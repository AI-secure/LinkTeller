import logging
import numpy as np
import scipy.sparse as sp
import time
import torch

from labeler import Labeler
from utils import feature_reader, get_noise, graph_reader, \
                normalize, sparse_mx_to_torch_sparse_tensor

class ClusterWorker():
    def __init__(self, args, dataset='', mode=''):
        self.args = args
        self.dataset = dataset
        self.mode = mode
        self.load_data()


    def build_cluster_adj(self, clean=False):
        adj = np.zeros((self.n_nodes, self.n_clusters), dtype=np.float64)

        for dst, src in self.edges.tolist():
            adj[src, self.fake_labels[dst]] += 1
            adj[dst, self.fake_labels[src]] += 1

        if self.mode in ( 'clusteradj' ) and not clean:
            adj += get_noise(
                self.args.noise_type, self.n_nodes, self.n_clusters, self.args.noise_seed,
                eps=self.args.epsilon, delta=self.args.delta)

            adj = np.clip(adj, a_min=0, a_max=None)
            adj = normalize(adj)
            return torch.FloatTensor(adj)

        adj = sp.coo_matrix(adj)
        adj = normalize(adj)
        return sparse_mx_to_torch_sparse_tensor(adj)


    def build_cluster_prj(self):
        unique, count = np.unique(self.fake_labels, return_counts=True)

        prj = np.zeros((self.n_clusters, self.n_nodes))

        for i, label in enumerate(self.fake_labels):
            prj[label, i] = 1 / count[label]
        return torch.FloatTensor(prj)


    def break_down(self):
        indice = [[] for i in range(self.n_classes)]
        for i, label in enumerate(self.fake_labels):
            indice[label].append(i)

        unique, count = torch.unique(self.fake_labels, return_counts=True)

        # print('unique', unique)
        # print('count', count)
        min_size = int(torch.min(count).item() * self.args.break_ratio + 0.5)
        if min_size == 0: min_size = 1
        # print('min_size', min_size)
        split = [self.labeler.get_equal_size(val, min_size) for val in count]

        # print('split', [elem[0] for elem in split], sum([elem[0] for elem in split]))

        t0 = time.time()
        start = 0

        for i in range(self.n_classes):
            idx = indice[i]
            if not idx: continue

            n_clusters, quota = split[(unique == i).nonzero().item()]

            self.fake_labels[idx] = self.labeler.get_cluster_labels(
                self.features[idx], n_clusters, quota=quota, start=start, same_size=True)
            start += n_clusters

        self.n_clusters = start

        print('generating broken down fake labels done using {} secs!'.format(time.time()-t0))
        # torch.save(self.fake_labels, 'flabels_{}.pt'.format(self.n_clusters))


    def build_adj_vanilla(self):
        adj = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float64)
        for dst, src in self.edges:
            adj[src][dst] = adj[dst][src] = 1

        t0 = time.time()
        adj += get_noise(self.args.noise_type, self.n_nodes, self.n_nodes, self.args.noise_seed, 
                            eps=self.args.epsilon, delta=self.args.delta)
        adj = np.clip(adj, a_min=0, a_max=None)
        print('adding noise done using {} secs!'.format(time.time() - t0))
        return adj


    def build_adj_original(self):
        adj = sp.coo_matrix((np.ones(self.edges.shape[0]), (self.edges[:, 0], self.edges[:, 1])),
                            shape=(self.n_nodes, self.n_nodes),
                            dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        return adj


    def build_adj_mat(self, mode='vanilla-clean'):
        if mode == 'vanilla-clean':
            adj = self.build_adj_original()

        elif mode == 'vanilla':
            adj = self.build_adj_vanilla()

        else:
            raise NotImplementedError('mode = {} not implemented!'.format(mode))

        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj) if mode == 'vanilla-clean' else torch.FloatTensor(adj)
        return adj


    def load_data(self):
        self.features, self.labels, self.idx_train, self.idx_val, self.idx_test \
            = feature_reader(dataset=self.dataset, scale=self.args.scale, 
                            train_ratio=self.args.train_ratio, feature_size=self.args.feature_size)

        # print('feature_size', self.features.shape)
        self.n_nodes = len(self.labels)
        self.n_features = self.features.shape[1]
        self.n_classes = self.labels.max().item() + 1

        self.edges = graph_reader(dataset=self.dataset)

        self.labeler = Labeler(self.features, self.labels, self.n_classes, 
                                self.idx_train, self.idx_val, self.idx_test)

        if self.mode in ( 'clusteradj', 'clusteradj-clean' ):
            self.generate_fake_labels()
            if self.args.break_down:
                self.break_down()
            self.adj = self.build_cluster_adj()
            self.prj = self.build_cluster_prj()
        else:
            self.adj = self.build_adj_mat(mode=self.mode)

        # self.calculate_connectivity()

        if torch.cuda.is_available():
            self.features = self.features.cuda()
            self.adj = self.adj.cuda()
            self.labels = self.labels.cuda()
            if hasattr(self, 'prj'):
                self.prj = self.prj.cuda()


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
        degrees = np.zeros(self.n_nodes)
        for edge in self.edges:
            u, v = edge
            degrees[u] += 1
            degrees[v] += 1
        return degrees


    def update_adj(self):
        if self.mode == 'clusteradj':
            self.adj = self.build_cluster_adj(clean=True)
        elif self.mode == 'vanilla':
            self.adj = self.build_adj_mat(mode='vanilla-clean')

        if torch.cuda.is_available():
            self.adj = self.adj.cuda()