from collections import defaultdict
import numpy as np
import os
import os.path as osp
from sklearn import metrics
import time
from tqdm import tqdm

import torch
import torch.nn.functional as F

from utils import construct_edge_sets, construct_edge_sets_from_random_subgraph, construct_edge_sets_through_bfs, construct_balanced_edge_sets

class Attacker:
    def __init__(self, args, model, worker):
        self.args = args
        self.dataset = args.dataset
        self.model = model
        self.worker = worker

        if args.sample_type == 'balanced-full':
            self.args.n_test = self.worker.n_nodes

        if self.dataset.startswith('twitch') or self.dataset.startswith('deezer'):
            self.features = self.worker.features_2
            self.adj = self.worker.adj_2

        else:
            self.features = self.worker.features
            self.adj = self.worker.adj_full


    def prepare_test_data(self):
        func = {
            'balanced': construct_edge_sets,
            'balanced-full': construct_balanced_edge_sets,
            'unbalanced': construct_edge_sets_from_random_subgraph,
            'unbalanced-lo': construct_edge_sets_from_random_subgraph,
            'unbalanced-hi': construct_edge_sets_from_random_subgraph,
            'bfs': construct_edge_sets_through_bfs,
        }.get(self.args.sample_type)
        if not func:
            raise NotImplementedError(f'sample_type = {self.args.sample_type} not implemented!')

        np.random.seed(self.args.sample_seed)
        (self.exist_edges, self.nonexist_edges), self.test_nodes = func(
            self.dataset, self.args.sample_type, self.worker.adj_ori, self.args.n_test)
        print(f'generating testing (non-)edge set done!')


    # \partial_f(x_u) / \partial_x_v
    def get_gradient(self, u, v):
        h = 0.0001
        ret = torch.zeros(self.worker.n_features)
        for i in range(self.worker.n_features):
            pert = torch.zeros_like(self.worker.features)
            pert[v][i] = h
            with torch.no_grad():
                grad = (self.model(self.worker.features + pert, self.worker.adj_full).detach() -
                        self.model(self.worker.features - pert, self.worker.adj_full).detach()) / (2 * h)
                ret[i] = grad[u].sum()

        return ret


    # # \partial_f(x_u) / \partial_epsilon_v
    # def get_gradient_eps(self, u, v):
    #     if self.dataset.startswith('twitch'):
    #         features = self.worker.features_2
    #         adj = self.worker.adj_2

    #     else:
    #         features = self.worker.features
    #         adj = self.worker.adj_full

    #     h = 0.00001
    #     pert_1 = torch.zeros_like(features)
    #     pert_2 = torch.zeros_like(features)

    #     pert_1[v] = features[v] * self.args.influence
    #     pert_2[v] = features[v] * h
    #     grad = (self.model(features + pert_1 + pert_2, adj).detach() -
    #             self.model(features + pert_1 - pert_2, adj).detach()) / (2 * h)

    #     return grad[u]


    # \partial_f(x_u) / \partial_epsilon_v
    def get_gradient_eps(self, u, v):
        pert_1 = torch.zeros_like(self.features)

        pert_1[v] = self.features[v] * self.args.influence

        grad = (self.model(self.features + pert_1, self.adj).detach() - 
                self.model(self.features, self.adj).detach()) / self.args.influence

        return grad[u]


    def get_gradient_eps_mat(self, v):
        pert_1 = torch.zeros_like(self.features)

        pert_1[v] = self.features[v] * self.args.influence

        grad = (self.model(self.features + pert_1, self.adj).detach() - 
                self.model(self.features, self.adj).detach()) / self.args.influence

        return grad


    def calculate_auc(self, v1, v0):
        v1 = sorted(v1)
        v0 = sorted(v0)
        vall = sorted(v1 + v0)

        TP = self.args.n_test
        FP = self.args.n_test
        T = F = self.args.n_test  # fixed

        p0 = p1 = 0

        TPR = TP / T
        FPR = FP / F

        result = [(FPR, TPR)]
        auc = 0
        for elem in vall:
            if p1 < self.args.n_test and abs(elem - v1[p1]) < 1e-6:
                p1 += 1
                TP -= 1
                TPR = TP / T
            else:
                p0 += 1
                FP -= 1
                FPR = FP / F
                auc += TPR * 1 / F

            result.append((FPR, TPR))

        return result, auc


    def link_prediction_attack(self):
        norm_exist = []
        norm_nonexist = []

        t = time.time()

        with torch.no_grad():
            for u, v in tqdm(self.exist_edges):

                grad = self.get_gradient_eps(u, v) # if self.args.approx else self.get_gradient(u, v)
                norm_exist.append(grad.norm().item())

            print(f'time for predicting existing edges: {time.time() - t}')

            t = time.time()
            for u, v in tqdm(self.nonexist_edges):

                grad = self.get_gradient_eps(u, v) # if self.args.approx else self.get_gradient(u, v)
                norm_nonexist.append(grad.norm().item())

            print(f'time for predicting non-existing edges: {time.time() - t}')

        # print(sorted(norm_exist))
        # print(sorted(norm_nonexist))


        y = [1] * len(norm_exist) + [0] * len(norm_nonexist)
        pred = norm_exist + norm_nonexist

        fpr, tpr, thresholds = metrics.roc_curve(y, pred)
        print('auc =', metrics.auc(fpr, tpr))

        precision, recall, thresholds_2 = metrics.precision_recall_curve(y, pred)
        print('ap =', metrics.average_precision_score(y, pred))

        folder_name = f'eval_{self.dataset}'
        if not osp.exists(folder_name): os.makedirs(folder_name)

        if self.args.mode == 'vanilla-clean':
            filename = osp.join(folder_name, f'{self.args.sample_type}_{self.args.n_test}_{self.args.sample_seed}.pt')
        else:
            filename = osp.join(folder_name, f'{self.args.sample_type}_{self.args.perturb_type}_{self.args.n_test}_{self.args.sample_seed}_eps-{self.args.epsilon}_seed-{self.args.noise_seed}.pt')

        torch.save({
            'auc': {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds
            },
            'pr': {
                'precision': precision,
                'recall': recall,
                'thresholds': thresholds_2
            },
            'result': {
                'y': y,
                'pred': pred,
            }
        }, filename)

        # result, auc = self.calculate_auc(norm_exist, norm_nonexist)

        # print('auc =', auc)
        # torch.save(result, 'result.pt')


    def link_prediction_attack_efficient(self):
        norm_exist = []
        norm_nonexist = []

        t = time.time()

        # 2. compute influence value for all pairs of nodes
        influence_val = np.zeros((self.args.n_test, self.args.n_test))

        with torch.no_grad():

            for i in tqdm(range(self.args.n_test)):
                u = self.test_nodes[i]
                grad_mat = self.get_gradient_eps_mat(u)

                for j in range(self.args.n_test):
                    v = self.test_nodes[j]

                    grad_vec = grad_mat[v]

                    influence_val[i][j] = grad_vec.norm().item()

            print(f'time for predicting edges: {time.time() - t}')

        node2ind = { node : i for i, node in enumerate(self.test_nodes) }

        for u, v in self.exist_edges:
            i = node2ind[u]
            j = node2ind[v]

            norm_exist.append(influence_val[j][i])

        for u, v in self.nonexist_edges:
            i = node2ind[u]
            j = node2ind[v]

            norm_nonexist.append(influence_val[j][i])

        self.compute_and_save(norm_exist, norm_nonexist)


    def link_prediction_attack_efficient_balanced(self):
        norm_exist = []
        norm_nonexist = []

        # organize exist_edges and nonexist_edges into dict
        edges_dict = defaultdict(list)
        nonedges_dict = defaultdict(list)
        for u, v in self.exist_edges:
            edges_dict[u].append(v)
        for u, v in self.nonexist_edges:
            nonedges_dict[u].append(v)

        t = time.time()
        with torch.no_grad():
            for u in tqdm(range(self.worker.n_nodes)):
                if u not in edges_dict and u not in nonedges_dict:
                    continue

                grad_mat = self.get_gradient_eps_mat(u)

                if u in edges_dict:
                    v_list = edges_dict[u]
                    for v in v_list:
                        grad_vec = grad_mat[v]
                        norm_exist.append(grad_vec.norm().item())

                if u in nonedges_dict:
                    v_list = nonedges_dict[u]
                    for v in v_list:
                        grad_vec = grad_mat[v]
                        norm_nonexist.append(grad_vec.norm().item())

            print(f'time for predicting edges: {time.time() - t}')

        self.compute_and_save(norm_exist, norm_nonexist)


    def baseline_attack(self):
        norm_exist = []
        norm_nonexist = []

        t = time.time()

        with torch.no_grad():
            # 0. compute posterior
            if self.args.attack_mode == 'baseline':
                if self.dataset != 'ppi':
                    posterior = F.softmax(self.model(self.features, self.adj), dim=1)
                else:
                    posterior = F.sigmoid(self.model(self.features, self.adj))
            elif self.args.attack_mode == 'baseline-feat':
                posterior = self.features
            else:
                raise NotImplementedError(f'attack_mode={self.args.attack_mode} not implemented!')

            # 1. compute the mean posterior of sampled nodes
            mean = torch.mean(posterior[self.test_nodes], dim=0)

            # 2. compute correlation value for all pairs of nodes
            dist = np.zeros((self.args.n_test, self.args.n_test))

            for i in tqdm(range(self.args.n_test)):
                u = self.test_nodes[i]
                for j in range(i+1, self.args.n_test):
                    v = self.test_nodes[j]

                    dist[i][j] = torch.dot(posterior[u] - mean, posterior[v] - mean) / torch.norm(posterior[u] - mean) / torch.norm(posterior[v] - mean)

            print(f'time for computing correlation value: {time.time() - t}')

        node2ind = { node : i for i, node in enumerate(self.test_nodes) }

        for u, v in self.exist_edges:
            i = node2ind[u]
            j = node2ind[v]

            norm_exist.append(dist[i][j] if i < j else dist[j][i])

        for u, v in self.nonexist_edges:
            i = node2ind[u]
            j = node2ind[v]

            norm_nonexist.append(dist[i][j] if i < j else dist[j][i])

        self.compute_and_save(norm_exist, norm_nonexist)


    def baseline_attack_balanced(self):
        norm_exist = []
        norm_nonexist = []

        # organize exist_edges and nonexist_edges into dict
        edges_dict = defaultdict(list)
        nonedges_dict = defaultdict(list)
        for u, v in self.exist_edges:
            edges_dict[u].append(v)
        for u, v in self.nonexist_edges:
            nonedges_dict[u].append(v)

        t = time.time()

        with torch.no_grad():
            # 0. compute posterior
            if self.args.attack_mode == 'baseline':
                if self.dataset != 'ppi':
                    posterior = F.softmax(self.model(self.features, self.adj), dim=1)
                else:
                    posterior = F.sigmoid(self.model(self.features, self.adj))
            elif self.args.attack_mode == 'baseline-feat':
                posterior = self.features
            else:
                raise NotImplementedError(f'attack_mode={self.args.attack_mode} not implemented!')

            # 1. compute the mean posterior of sampled nodes
            mean = torch.mean(posterior, dim=0)

            # 2. compute correlation value for all pairs
            for u, v in tqdm(self.exist_edges):
                norm_exist.append((torch.dot(posterior[u] - mean, posterior[v] - mean) / torch.norm(posterior[u] - mean) / torch.norm(posterior[v] - mean)).item())

            for u, v in tqdm(self.nonexist_edges):
                norm_nonexist.append((torch.dot(posterior[u] - mean, posterior[v] - mean) / torch.norm(posterior[u] - mean) / torch.norm(posterior[v] - mean)).item())

            print(f'time for computing correlation value: {time.time() - t}')

        self.compute_and_save(norm_exist, norm_nonexist)


    def compute_and_save(self, norm_exist, norm_nonexist):
        y = [1] * len(norm_exist) + [0] * len(norm_nonexist)
        pred = norm_exist + norm_nonexist

        fpr, tpr, thresholds = metrics.roc_curve(y, pred)
        print('auc =', metrics.auc(fpr, tpr))

        precision, recall, thresholds_2 = metrics.precision_recall_curve(y, pred)
        print('ap =', metrics.average_precision_score(y, pred))

        folder_name = f'eval_{self.dataset}'
        if not osp.exists(folder_name): os.makedirs(folder_name)

        if self.args.mode == 'vanilla-clean':
            filename = osp.join(folder_name, f'{self.args.attack_mode}_{self.args.sample_type}_{self.args.n_test}_{self.args.sample_seed}.pt')
        else:
            filename = osp.join(folder_name, f'{self.args.attack_mode}_{self.args.sample_type}_{self.args.perturb_type}_{self.args.n_test}_{self.args.sample_seed}_eps-{self.args.epsilon}_seed-{self.args.noise_seed}.pt')

        torch.save({
            'auc': {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds
            },
            'pr': {
                'precision': precision,
                'recall': recall,
                'thresholds': thresholds_2
            },
            'result': {
                'y': y,
                'pred': pred,
            }
        }, filename)
        print(f'attack results saved to: {filename}')

        # result, auc = self.calculate_auc(norm_exist, norm_nonexist)

        # print('auc =', auc)
        # torch.save(result, 'result.pt')
