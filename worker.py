import logging
import math
import numpy as np
import random
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
import time
from tqdm import tqdm

import torch
from torch.nn.parameter import Parameter

from utils import feature_reader, get_noise, graph_reader, \
                normalize, sparse_mx_to_torch_sparse_tensor, t_normalize, fetch_normalization, \
                compressive_sensing

class Worker():
    def __init__(self, args, dataset='', mode=''):
        self.args = args
        self.dataset = dataset
        self.mode = mode
        self.transfer = (not self.dataset.startswith('twitch-train') and self.dataset.startswith('twitch')) or \
                        self.dataset.startswith('wikipedia') or \
                        self.dataset.startswith('deezer')
        self.load_data()


    def build_cluster_adj(self, clean=False, fnormalize=True):
        adj = np.zeros((self.n_nodes, self.n_clusters), dtype=np.float64)

        for dst, src in self.edges.tolist():
            adj[src, self.fake_labels[dst]] += 1
            adj[dst, self.fake_labels[src]] += 1

        if self.mode in ( 'clusteradj' ) and not clean:
            adj += get_noise(
                self.args.noise_type, size=self.n_nodes*self.n_clusters, seed=self.args.noise_seed,
                eps=self.args.epsilon, delta=self.args.delta).reshape(self.n_nodes, self.n_clusters)

            adj = np.clip(adj, a_min=0, a_max=None)

            if fnormalize:
                adj = normalize(adj)
            else:
                adj = normalize(np.dot(adj, self.prj.cpu().numpy()) + np.eye(self.n_nodes))

            return torch.FloatTensor(adj)

        # adj = sp.coo_matrix(adj)
        if fnormalize:
            adj = normalize(adj)
        elif self.mode != 'degree_nlp':
            adj = normalize(np.dot(adj, self.prj.cpu().numpy()) + np.eye(self.n_nodes))

        # adj = sparse_mx_to_torch_sparse_tensor(adj)

        # if not fnormalize and not self.mode == 'degree_mlp':
        #     adj = t_normalize(torch.mm(adj, self.prj) + torch.eye(self.n_nodes))

        # return adj
        return torch.FloatTensor(adj)


    def build_cluster_prj(self):
        unique, count = np.unique(self.fake_labels.cpu(), return_counts=True)

        count_dict = {k:v for k, v in zip(unique, count)}

        prj = np.zeros((self.n_clusters, self.n_nodes))

        for i, label in enumerate(self.fake_labels):
            label = label.item()
            prj[label, i] = 1 / count_dict[label]

        # prj = sp.coo_matrix(prj)
        # return sparse_mx_to_torch_sparse_tensor(prj)
        return torch.FloatTensor(prj)


    def build_degree_vec(self):
        vec = np.zeros((self.n_nodes, 1))
        for u, v in self.edges:
            vec[u][0] += 1
            vec[v][0] += 1

        return torch.FloatTensor(vec)


    def break_down(self):
        unique, count = torch.unique(self.fake_labels, return_counts=True)
        indice = [(self.fake_labels == x).nonzero().squeeze() for x in unique]

        if self.args.break_method == 'kmeans':

            min_size = int(torch.min(count).item() * self.args.break_ratio + 0.5)
            if min_size in (0, 1):
                print('skip the step of generating broken labels!')

            else:

                # print('min_size', min_size)
                split = [self.labeler.get_equal_size(val, min_size) for val in count]

                # print('split', split)

                t0 = time.time()
                start = 0

                for idx, (n_clusters, quota) in zip(indice, split):
                    self.fake_labels[idx] = self.labeler.get_cluster_labels(
                        self.features[idx].cpu(), n_clusters, quota=quota, start=start, same_size=True).cuda()
                    start += n_clusters

                self.n_clusters = start

                print('generating broken down fake labels done using {} secs!'.format(time.time()-t0))
            # torch.save(self.fake_labels, 'flabels_{}.pt'.format(self.n_clusters))

        else:
            print(f'break_method = {self.args.break_method} not implemented!')
            exit(-1)

        logging.info(f'num_cluster = {self.n_clusters}')


    def build_adj_vanilla(self):
        adj = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float64)
        for dst, src in self.edges:
            adj[src][dst] = adj[dst][src] = 1

        t0 = time.time()
        # adj += get_noise(self.args.noise_type, size=self.n_nodes*self.n_nodes, seed=self.args.noise_seed, 
        #                     eps=self.args.epsilon, delta=self.args.delta).reshape(self.n_nodes, self.n_nodes)
        # adj = np.clip(adj, a_min=0, a_max=None)

        s = 2 / (np.exp(self.args.epsilon) + 1)
        print(f's={s:.4f}')
        bernoulli = np.random.binomial(1, s, self.n_nodes * (self.n_nodes-1))
        entry = np.where(bernoulli)
        for u, v in zip(*entry):
            if u >= v: continue
            x = np.random.binomial(1, 0.5)
            adj[u][v] = adj[v][u] = x

        print('adding noise done using {} secs!'.format(time.time() - t0))
        return adj


    # def calc_index(self, k):
    #     lo = (-1 + math.sqrt(8 * k + 9)) / 2
    #     # hi = (1 + math.sqrt(8 * k + 1)) / 2
    #     if lo - int(lo) < 1e-6:
    #         i = int(lo)
    #     else:
    #         i = int(lo) + 1
    #     j = k - (i-1) * i // 2
    #     assert(j <= i-1)
    #     assert(i < self.n_nodes and j < self.n_nodes), f'{k}, {i}, {j} invalid!'

    #     return i, j


    def calc_index(self, N, k):
        lo = (2*N-3 - math.sqrt((2*N-3)**2 - 4*(2*k-2*N+4))) / 2
        # hi = (1 + math.sqrt(8 * k + 1)) / 2
        if lo - int(lo) < 1e-6:
            i = int(lo)
        else:
            i = int(lo) + 1
            
        j = k - (2*N-1-i)*i//2 + i + 1

        assert(j > i and j < N and i < N)

        return i, j


    def construct_sparse_mat(self, indice, N):
        cur_row = -1
        new_indices = []
        new_indptr = []

        for i, j in tqdm(indice):
            if i >= j:
                continue

            while i > cur_row:
                new_indptr.append(len(new_indices))
                cur_row += 1

            new_indices.append(j)

        while N > cur_row:
            new_indptr.append(len(new_indices))
            cur_row += 1

        data = np.ones(len(new_indices), dtype=np.int64)
        indices = np.asarray(new_indices, dtype=np.int64)
        indptr = np.asarray(new_indptr, dtype=np.int64)

        mat = sp.csr_matrix((data, indices, indptr), (N, N))

        return mat + mat.T


    def perturb_adj(self, adj, perturb_type):
        if perturb_type == 'discrete':
            return self.perturb_adj_discrete(adj)
        else:
            return self.perturb_adj_continuous(adj)


    def perturb_adj_discrete(self, adj):
        s = 2 / (np.exp(self.args.epsilon) + 1)
        print(f's = {s:.4f}')
        N = adj.shape[0]

        t = time.time()
        # bernoulli = np.random.binomial(1, s, N * (N-1) // 2)
        # entry = np.where(bernoulli)[0]

        np.random.seed(self.args.noise_seed)
        bernoulli = np.random.binomial(1, s, (N, N))
        print(f'generating perturbing vector done using {time.time() - t} secs!')
        logging.info(f'generating perturbing vector done using {time.time() - t} secs!')
        entry = np.asarray(list(zip(*np.where(bernoulli))))

        dig_1 = np.random.binomial(1, 1/2, len(entry))
        indice_1 = entry[np.where(dig_1 == 1)[0]]
        indice_0 = entry[np.where(dig_1 == 0)[0]]

        add_mat = self.construct_sparse_mat(indice_1, N)
        minus_mat = self.construct_sparse_mat(indice_0, N)

        # # add_mat = np.zeros_like(adj.A)
        # add_row = []
        # add_col = []
        # # minus_mat = np.zeros_like(adj.A)
        # minus_row = []
        # minus_col = []

        # for i in tqdm(range(N)):
        #     for j in range(i+1, N):
        #         x = np.random.binomial(1, s, 1)
        #         if x == 1:
        #             x = np.random.binomial(1, 1/2, 1)
        #             if x == 1:
        #                 # add_mat[i, j] = x
        #                 add_row.append(i)
        #                 add_col.append(j)
        #             else:
        #                 # minus_mat[i, j] = x
        #                 minus_row.append(i)
        #                 minus_col.append(j)
        # add_data = np.ones(len(add_row), dtype=np.int32)
        # minus_data = np.ones(len(minus_row), dtype=np.int32)
        # # add_mat = sp.csr_matrix(add_mat)
        # # minus_mat = sp.csr_matrix(minus_mat)
        # add_mat = sp.csr_matrix(add_data, (add_row, add_col))
        # minus_mat = sp.csr_matrix(minus_data, (minus_row, minus_col))
        # add_mat += add_mat.T
        # minus_mat += minus_mat.T

        adj_noisy = adj + add_mat - minus_mat

        adj_noisy.data[np.where(adj_noisy.data == -1)[0]] = 0
        adj_noisy.data[np.where(adj_noisy.data == 2)[0]] = 1

        # adj = sp.lil_matrix(adj)
        # for k in tqdm(indice_1):
        #     i, j = self.calc_index(k)
        #     adj[i, j] = adj[j, i] = 1

        # for k in tqdm(indice_0):
        #     i, j = self.calc_index(k)
        #     adj[i, j] = adj[j, i] = 0

        return adj_noisy


    def perturb_adj_continuous(self, adj):
        self.n_nodes = adj.shape[0]
        n_edges = len(adj.data) // 2

        N = self.n_nodes
        t = time.time()

        A = sp.tril(adj, k=-1)
        print('getting the lower triangle of adj matrix done!')

        eps_1 = self.args.epsilon * 0.01
        eps_2 = self.args.epsilon - eps_1
        noise = get_noise(noise_type=self.args.noise_type, size=(N, N), seed=self.args.noise_seed, 
                        eps=eps_2, delta=self.args.delta, sensitivity=1)
        noise *= np.tri(*noise.shape, k=-1, dtype=np.bool)
        print(f'generating noise done using {time.time() - t} secs!')

        A += noise
        print(f'adding noise to the adj matrix done!')

        t = time.time()
        n_edges_keep = n_edges + int(
            get_noise(noise_type=self.args.noise_type, size=1, seed=self.args.noise_seed, 
                    eps=eps_1, delta=self.args.delta, sensitivity=1)[0])
        print(f'edge number from {n_edges} to {n_edges_keep}')

        t = time.time()
        a_r = A.A.ravel()

        n_splits = 50
        len_h = len(a_r) // n_splits
        ind_list = []
        for i in tqdm(range(n_splits - 1)):
            ind = np.argpartition(a_r[len_h*i:len_h*(i+1)], -n_edges_keep)[-n_edges_keep:]
            ind_list.append(ind + len_h * i)

        ind = np.argpartition(a_r[len_h*(n_splits-1):], -n_edges_keep)[-n_edges_keep:]
        ind_list.append(ind + len_h * (n_splits - 1))

        ind_subset = np.hstack(ind_list)
        a_subset = a_r[ind_subset]
        ind = np.argpartition(a_subset, -n_edges_keep)[-n_edges_keep:]

        row_idx = []
        col_idx = []
        for idx in ind:
            idx = ind_subset[idx]
            row_idx.append(idx // N)
            col_idx.append(idx % N)
            assert(col_idx < row_idx)
        data_idx = np.ones(n_edges_keep, dtype=np.int32)
        print(f'data preparation done using {time.time() - t} secs!')

        mat = sp.csr_matrix((data_idx, (row_idx, col_idx)), shape=(N, N))
        return mat + mat.T


    def build_adj_original(self, edges):
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(self.n_nodes, self.n_nodes),
                            dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        return adj


    def build_adj_mat(self, edges, mode='vanilla-clean'):
        if mode in ( 'vanilla-clean', 'degcn-clean' ):
            adj = self.build_adj_original(edges)

        elif mode in ( 'vanilla', 'degcn' ):
            adj = self.build_adj_vanilla()
            if mode == 'degcn':
                # temp = np.zeros((self.n_nodes, self.n_nodes))
                # temp[adj > 0.5] = 1
                # adj = temp

                # print(len(self.edges))
                self.edges = []
                for u, v in zip(*np.where(adj)):
                    if u > v: continue
                    self.edges.append((u, v))

                print(len(self.edges))

        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj) if mode in ( 'vanilla-clean', 'degcn-clean' ) else torch.FloatTensor(adj)
        return adj


    def sgc_precompute(self, adj, features, mode='sgc-clean'):
        # # if mode == 'sgc-clean':
        # adj = self.build_adj_original()
        # # else:
        # #     adj = self.build_adj_vanilla()

        normalizer = fetch_normalization(self.args.norm)
        adj = sparse_mx_to_torch_sparse_tensor(normalizer(adj)).float().cuda()

        # adj_normalizer = fetch_normalization(self.args.normalization)
        # adj = adj_normalizer(adj)
        # adj = sparse_mx_to_torch_sparse_tensor(adj).float().cuda()

        # for _ in range(self.args.degree):
        features = torch.spmm(adj, features)

        return features


    def decompose_graph(self):
        self.sub_adj = []
        num = len(self.edges)
        for i in range(3):
            sub_edges = [self.edges[i] for i in range(i, num, 3)]
            self.sub_adj.append(self.build_adj_mat(np.asarray(sub_edges), mode='degcn-clean'))


    def construct_hop_dict(self):
        self.edge_dict = {u : set() for u in range(self.n_nodes)}
        for u, v in self.edges:
            self.edge_dict[u].add(v)
            self.edge_dict[v].add(u)

        self.two_hop_dict = {u : set() for u in range(self.n_nodes)}

        self.one_hop_edges = []
        for u in self.edge_dict:
            for v in self.edge_dict[u]:
                for p in self.edge_dict[v]:
                    if p > u and u not in self.edge_dict[p]:
                        self.one_hop_edges.append((u, p))
                        self.two_hop_dict[u].add(p)
                        self.two_hop_dict[p].add(u)

        self.two_hop_edges = []
        for u in self.edge_dict:
            for v in self.edge_dict[u]:
                for p in self.two_hop_dict[v]:
                    if p > u and u not in self.edge_dict[p] and u not in self.two_hop_dict[p]:
                        self.two_hop_edges.append((u, p))


    def load_data(self):
        if self.dataset in ( 'reddit', 'flickr', 'ppi', 'ppi-large', 'cora', 'citeseer', 'pubmed' ):
            self.features, self.features_train, self.labels, self.idx_train, self.idx_val, self.idx_test \
                = feature_reader(dataset=self.dataset, scale=self.args.scale, 
                                train_ratio=self.args.train_ratio, feature_size=self.args.feature_size)

            if torch.cuda.is_available():
                self.features = self.features.cuda()
                self.features_train = self.features_train.cuda()
                self.labels = self.labels.cuda()

            self.n_nodes = len(self.labels)
            self.n_features = self.features.shape[1]
            self.multi_label = self.labels.shape[1]
            if self.multi_label == 1:
                self.n_classes = self.labels.max().item() + 1
            else:
                self.n_classes = self.multi_label

        elif self.dataset.startswith( 'twitch-train' ):
            p = self.dataset.find('/')
            self.features, self.labels = feature_reader(dataset=f'twitch/{self.dataset[p+1:]}')
            self.n_nodes = len(self.labels)
            self.n_nodes_1 = int(0.8 * self.n_nodes)
            self.n_nodes_2 = self.n_nodes - self.n_nodes_1
            self.idx_train = np.random.choice(self.n_nodes, self.n_nodes_1, replace=False)
            self.idx_val = np.asarray( list( set(range(self.n_nodes)) - set(range(self.n_nodes_1)) ) )

            self.features_train = self.features[self.idx_train]

            scaler = StandardScaler()
            scaler.fit(self.features_train)
            self.features = scaler.transform(self.features)
            self.features = torch.FloatTensor(self.features)
            self.features_train = self.features[self.idx_train]

            if torch.cuda.is_available():
                self.features = self.features.cuda()
                self.features_train = self.features_train.cuda()
                self.labels = self.labels.cuda()

            self.n_features = 3170
            self.multi_label = 1
            self.n_classes = 2


        elif self.dataset.startswith( 'twitch' ):
            p_0 = self.dataset.find('/')
            data_folder = self.dataset[:p_0]

            p = self.dataset.rfind('/')+1
            self.dataset1 = self.dataset[:p-1]
            self.dataset2 = f'{data_folder}/{self.dataset[p:]}'

            self.features_1, self.labels_1 = feature_reader(dataset=self.dataset1)
            self.features_2, self.labels_2 = feature_reader(dataset=self.dataset2)

            scaler = StandardScaler()
            scaler.fit(self.features_1)
            self.features_1 = torch.FloatTensor(scaler.transform(self.features_1))
            self.features_2 = torch.FloatTensor(scaler.transform(self.features_2))

            if torch.cuda.is_available():
                self.features_1 = self.features_1.cuda()
                self.features_2 = self.features_2.cuda()
                self.labels_1 = self.labels_1.cuda()
                self.labels_2 = self.labels_2.cuda()

            self.n_nodes_1 = len(self.labels_1)
            self.n_nodes_2 = len(self.labels_2)
            self.n_features = 3170
            self.multi_label = 1
            self.n_classes = 2

        elif self.dataset.startswith( 'deezer' ):
            p_0 = self.dataset.find('/')
            data_folder = self.dataset[:p_0]

            p = self.dataset.rfind('/')+1
            self.dataset1 = self.dataset[:p-1]
            self.dataset2 = f'{data_folder}/{self.dataset[p:]}'

            self.labels_1 = feature_reader(dataset=self.dataset1)
            self.labels_2 = feature_reader(dataset=self.dataset2)

            if torch.cuda.is_available():
                self.labels_1 = self.labels_1.cuda()
                self.labels_2 = self.labels_2.cuda()

            self.n_nodes_1 = len(self.labels_1)
            self.n_nodes_2 = len(self.labels_2)
            self.n_classes = self.multi_label = 84

        else:
            raise NotImplementedError(f'dataset = {self.dataset} not implemented!')

        print(f'loading {self.dataset} features done!')

        # print('feature_size', self.features.shape)

        # print('====================================')
        # print('||   n_nodes =', self.n_nodes)
        # print('||   n_features =', self.n_features)
        # print('||   n_classes =', self.n_classes, '(', self.multi_label, ')')
        # print('====================================')

        if self.args.mode in ( 'mlp', 'lr' ): return

        if self.dataset in ( 'reddit', 'flickr', 'ppi', 'ppi-large', 'cora', 'citeseer', 'pubmed' ):
            self.adj_full = graph_reader(args=self.args, dataset=self.dataset, n_nodes=self.n_nodes)

            # construct training data
            if self.dataset in ( 'cora', 'citeseer', 'pubmed' ):
                self.adj_train = sp.csr_matrix.copy(self.adj_full)
                self.adj_ori = sp.csr_matrix.copy(self.adj_full)
            else:
                self.adj_train = self.adj_full[self.idx_train, :][:, self.idx_train]
                self.adj_ori = sp.csr_matrix.copy(self.adj_full)

        elif self.dataset.startswith( 'twitch-train' ):
            p = self.dataset.find('/')
            self.adj_full = graph_reader(args=self.args, dataset=f'twitch/{self.dataset[p+1:]}', n_nodes=self.n_nodes)
            self.adj_train = self.adj_full[self.idx_train, :][:, self.idx_train]
            self.adj_ori = sp.csr_matrix.copy(self.adj_full)

        elif self.dataset.startswith( 'twitch' ):
            self.adj_1 = graph_reader(args=self.args, dataset=self.dataset1, n_nodes=self.n_nodes_1)
            self.adj_2 = graph_reader(args=self.args, dataset=self.dataset2, n_nodes=self.n_nodes_2)
            self.adj_ori = sp.csr_matrix.copy(self.adj_2)

        elif self.dataset.startswith( 'deezer' ):
            self.adj_1, self.features_1 = graph_reader(args=self.args, dataset=self.dataset1, n_nodes=self.n_nodes_1)
            self.adj_2, self.features_2 = graph_reader(args=self.args, dataset=self.dataset2, n_nodes=self.n_nodes_2)
            self.adj_ori = sp.csr_matrix.copy(self.adj_2)
            self.n_features = self.features_1.shape[-1]

            if torch.cuda.is_available():
                self.features_1 = self.features_1.cuda()
                self.features_2 = self.features_2.cuda()

        else:
            self.edges = graph_reader(args=self.args, dataset=self.dataset)

        # self.construct_hop_dict()

        # self.exist_edges = random.sample(self.edges.tolist(), self.n_test)
        # self.nonexist_edges = random.sample(self.one_hop_edges, self.n_test)

        # self.nonexist_edges = random.sample(self.two_hop_edges, self.n_test)
        # self.nonexist_edges = random.sample(self.two_hop_edges+self.one_hop_edges, self.n_test)
        # self.nonexist_edges = []
        # cnt_nonexist = 0
        # while 1:
        #     u = np.random.choice(self.n_nodes)
        #     v = np.random.choice(self.n_nodes)
        #     if u != v and v not in self.edge_dict[u]:
        #         self.nonexist_edges.append((u, v))
        #         cnt_nonexist += 1
        #     if cnt_nonexist == self.n_test: break

        # self.labeler = Labeler(self.features, self.labels, self.n_classes, 
        #                         self.idx_train, self.idx_val, self.idx_test)

        self.prepare_data()

    def prepare_data(self):
        if self.mode in ( 'sgc-clean', 'sgc' ):
            if self.dataset in ( 'reddit', 'flickr', 'ppi', 'ppi-large', 'cora', 'citeseer', 'pubmed' ):
                self.features_train = self.sgc_precompute(self.adj_train, self.features_train, mode=self.mode)
                self.features = self.sgc_precompute(self.adj_full, self.features, mode=self.mode)
                self.adj = self.adj_train = None

            elif self.transfer:
                self.features_1 = self.sgc_precompute(self.adj_1, self.features_1, mode=self.mode)
                self.features_2 = self.sgc_precompute(self.adj_2, self.features_2, mode=self.mode)
                self.adj_1 = self.adj_2 = None

            else:
                raise NotImplementedError(f'dataset = {self.dataset} not implemented!')

            print('SGC Precomputing done!')

        elif self.mode in ( 'clusteradj', 'clusteradj-clean' ):
            self.generate_fake_labels()
            if self.args.break_down:
                self.break_down()

            self.prj = self.build_cluster_prj()
            self.adj = self.build_cluster_adj(fnormalize=self.args.fnormalize)

        elif self.mode in ( 'vanilla', 'vanilla-clean', 'cs' ):
            if self.dataset in ( 'reddit', 'flickr', 'ppi', 'ppi-large', 'cora', 'citeseer', 'pubmed' ) \
                or self.dataset.startswith('twitch-train'):
                if self.mode == 'vanilla':
                    self.adj_full = self.perturb_adj(self.adj_full, self.args.perturb_type)
                    self.adj_train = self.perturb_adj(self.adj_train, self.args.perturb_type)
                    print('perturbing done!')

                # normalize adjacency matrix
                if self.dataset not in ( 'cora', 'citeseer', 'pubmed' ):
                    normalizer = fetch_normalization(self.args.norm)
                    self.adj_train = normalizer(self.adj_train)
                    self.adj_full = normalizer(self.adj_full)

                self.adj_train = sparse_mx_to_torch_sparse_tensor(self.adj_train)
                self.adj_full = sparse_mx_to_torch_sparse_tensor(self.adj_full)

            elif self.transfer:
                if self.mode == 'vanilla':
                    self.adj_1 = self.perturb_adj(self.adj_1, self.args.perturb_type)
                    self.adj_2 = self.perturb_adj(self.adj_2, self.args.perturb_type)
                    print('perturbing done!')

                elif self.mode == 'cs':
                    self.adj_1 = compressive_sensing(self.args, self.adj_1)
                    self.adj_2 = compressive_sensing(self.args, self.adj_2)
                    print('compressive sensing done!')

                # normalize adjacency matrix
                normalizer = fetch_normalization(self.args.norm)
                self.adj_1 = sparse_mx_to_torch_sparse_tensor(normalizer(self.adj_1))
                self.adj_2 = sparse_mx_to_torch_sparse_tensor(normalizer(self.adj_2))

            else:
                # self.adj = self.build_adj_mat(self.edges, mode=self.mode)
                raise NotImplementedError(f'dataset = {self.dataset} not implemented!')

            print('Normalizing Adj done!')

        elif self.mode in ( 'degree_mlp', 'basic_mlp' ):
            self.adj = None

        elif self.mode in ( 'degcn', 'degcn-clean' ):
            self.adj = self.build_adj_mat(self.edges, mode=self.mode)
            self.decompose_graph()

        else:
            raise NotImplementedError('mode = {} not implemented!'.format(self.mode))

        # self.calculate_connectivity()

        if torch.cuda.is_available():
            if hasattr(self, 'adj') and self.adj is not None:
                self.adj = self.adj.cuda()
            if hasattr(self, 'adj_train') and self.adj_train is not None:
                self.adj_train = self.adj_train.cuda()
                self.adj_full = self.adj_full.cuda()
            if hasattr(self, 'adj_1') and self.adj_1 is not None:
                self.adj_1 = self.adj_1.cuda()
                self.adj_2 = self.adj_2.cuda()
            if hasattr(self, 'prj'):
                self.prj = self.prj.cuda()
            if hasattr(self, 'sub_adj'):
                for i in range(len(self.sub_adj)):
                    self.sub_adj[i] = self.sub_adj[i].cuda()


    def generate_fake_labels(self):
        cluster_method = self.args.cluster_method
        t0 = time.time()

        if cluster_method == 'random':
            self.n_clusters = self.args.n_clusters
            self.fake_labels = self.labeler.get_random_labels(self.n_clusters, self.args.cluster_seed)

        elif cluster_method == 'hierarchical':
            init_method = self.args.init_method
            self.n_clusters = self.n_classes

            if init_method == 'naive':
                self.fake_labels = self.labeler.get_naive_labels(self.args.assign_seed)

            elif init_method == 'voting':
                self.fake_labels = self.labeler.get_majority_labels(self.edges, self.args.assign_seed)

            elif init_method == 'knn':
                self.fake_labels = self.labeler.get_knn_labels(self.args.knn)

            elif init_method == 'gt':
                self.fake_labels = self.labels.clone()

            else:
                raise NotImplementedError('init_method={} in cluster_method=label not implemented!'.format(init_method))

        elif cluster_method in ( 'kmeans', 'sskmeans' ):
            self.n_clusters = self.args.n_clusters
            self.fake_labels = self.labeler.get_kmeans_labels(
                self.n_clusters, self.args.knn, cluster_method, same_size=self.args.same_size)

        else:
            raise NotImplementedError('cluster_method={} not implemented!'.format(cluster_method))

        print('generating fake labels done using {} secs!'.format(time.time()-t0))
        # torch.save(self.fake_labels, 'flabels_{}.pt'.format(self.n_clusters))


    def calculate_connectivity(self):
        n_edges = len(self.edges)
        kappa = n_edges / (0.5*self.n_nodes*(self.n_nodes-1))
        labels = self.fake_labels

        edge_adj = np.zeros((self.n_clusters, self.n_clusters))
        for edge in self.edges:
            u, v = labels[edge[0]], labels[edge[1]]
            edge_adj[u][v] += 1
            edge_adj[v][u] += 1

        unique, count = np.unique(labels, return_counts=True)

        kappa_intra = 0
        for i in range(self.n_clusters):
            kappa_intra += edge_adj[i][i] / (0.5 * count[i] * (count[i]-1))
        kappa_intra /= self.n_clusters

        kappa_inter = 0
        for i in range(self.n_clusters):
            for j in range(i+1, self.n_clusters):
                kappa_inter += edge_adj[i][j] / (count[i] * count[j])
        kappa_inter /= (0.5 * self.n_clusters * (self.n_clusters - 1))

        print('k_inter = {:4f}, k = {:4f}, k_intra = {:4f}'.format(kappa_inter, kappa, kappa_intra))
        logging.info('k_inter = {:4f}, k = {:4f}, k_intra = {:4f}'.format(kappa_inter, kappa, kappa_intra))


    def calculate_degree(self):
        if self.dataset.startswith( 'twitch' ):
            degrees = np.zeros(self.n_nodes_2)
            adj = self.adj_2
        else:
            degrees = np.zeros(self.n_nodes)
            adj = self.adj_train

        self.edges = []
        for u, v in zip(*np.where(adj.cpu().to_dense())):
            if u > v: continue
            degrees[u] += 1
            degrees[v] += 1
        return degrees


    def update_adj(self):
        if self.mode == 'clusteradj':
            self.adj = self.build_cluster_adj(clean=True, fnormalize=self.args.fnormalize)

        elif self.mode == 'vanilla':
            self.adj = self.build_adj_mat(self.edges)

        elif self.mode == 'sgc':
            self.features = self.sgc_precompute()

        elif self.mode == 'degcn':
            pass

        if torch.cuda.is_available():
            self.adj = self.adj.cuda()