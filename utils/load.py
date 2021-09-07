from collections import OrderedDict, defaultdict
import itertools
import json
import math
import networkx as nx
import numpy as np
import os
import pandas as pd
import pickle as pkl
from queue import Queue
import random
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import torch
from tqdm import tqdm

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def get_noise(noise_type, size, seed, eps=10, delta=1e-5, sensitivity=2):
    np.random.seed(seed)

    if noise_type == 'laplace':
        noise = np.random.laplace(0, sensitivity/eps, size)
    elif noise_type == 'gaussian':
        c = np.sqrt(2*np.log(1.25/delta))
        stddev = c * sensitivity / eps
        noise = np.random.normal(0, stddev, size)
    else:
        raise NotImplementedError('noise {} not implemented!'.format(noise_type))

    return noise


def feature_reader(dataset="cora", scale='large', train_ratio=0.5, feature_size=-1):
    if dataset.startswith('twitch/') or dataset.startswith('wikipedia/') \
            or dataset.startswith('facebook'):
        if dataset.startswith('twitch') or dataset.startswith('wikipedia/'):
            identifier = dataset[dataset.find('/')+1:]
            filename = './data/{}/musae_{}_features.json'.format(dataset, identifier)
        
        else:
            filename = './data/facebook/musae_facebook_features.json'
        with open(filename) as f:
            data = json.load(f)
            n_nodes = len(data)

            items = sorted(set(itertools.chain.from_iterable(data.values())))
            n_features = 3170 if dataset.startswith('twitch') else max(items) + 1

            features = np.zeros((n_nodes, n_features))
            for idx, elem in data.items():
                features[int(idx), elem] = 1

        if dataset.startswith('twitch'):
            data = pd.read_csv('./data/{}/musae_{}_target.csv'.format(dataset, identifier))
            mature = list(map(int, data['mature'].values))
            new_id = list(map(int, data['new_id'].values))
            idx_map = {elem : i for i, elem in enumerate(new_id)}
            labels = [mature[idx_map[idx]] for idx in range(n_nodes)]
        elif dataset.startswith('wikipedia/'):
            data = pd.read_csv('./data/{}/musae_{}_target.csv'.format(dataset, identifier))
            labels = list(map(int, data['target'].values))
        else:
            data = pd.read_csv('./data/facebook/musae_facebook_target.csv')
            labels = data['page_type'].values.tolist()
            all_labels = sorted(set(labels))
            label_dict = {label: i for i, label in enumerate(all_labels)}
            labels = list(map(label_dict.get, labels))

        labels = torch.LongTensor(labels)

        unique, count = torch.unique(labels, return_counts=True)

        # len_all = len(labels)
        # len_valid = len_test = int(0.1 * len_all)
        # len_train = int(train_ratio * len_all)
        # idx_all = list(range(len_all))

        # random.seed(42)
        # random.shuffle(idx_all)
        # idx_test = idx_all[:len_test]
        # idx_valid = idx_all[len_test:len_test+len_valid]
        # idx_train = idx_all[len_test+len_valid:len_test+len_valid+len_train]

        return features, labels

    elif dataset.startswith('deezer'):
        identifier = dataset[dataset.find('/')+1:]
        filename = './data/deezer/{}_genres.json'.format(identifier)
        with open(filename) as f:
            data = json.load(f)
            n_nodes = len(data)

            items = sorted(set(itertools.chain.from_iterable(data.values())))
            item_mapping = {item : i for i, item in enumerate(items)}
            n_classes = 84

            labels = np.zeros((n_nodes, n_classes))
            for idx, elem in data.items():
                elem_int = np.asarray(list(map(item_mapping.get, elem)))
                labels[int(idx), elem_int] = 1

        labels = torch.LongTensor(labels)
        return labels


    elif dataset in ('reddit', 'flickr', 'ppi', 'ppi-large'):
        # role
        role = json.load(open(f'./data/{dataset}/role.json'))
        idx_train = np.asarray(sorted(role['tr']))
        idx_valid = np.asarray(sorted(role['va']))
        idx_test = np.asarray(sorted(role['te']))

        # features
        features = np.load(f'./data/{dataset}/feats.npy')
        features_train = features[idx_train]

        scaler = StandardScaler()
        scaler.fit(features_train)
        features = scaler.transform(features)
        features = torch.FloatTensor(features)
        features_train = features[idx_train]

        n_nodes = len(features)

        # label
        class_map = json.load(open(f'./data/{dataset}/class_map.json'))

        multi_label = 1
        for key, value in class_map.items():
            if type(value) == list:
                multi_label = len(value)    # single-label vs multi-label
            break

        labels = np.zeros((n_nodes, multi_label))
        for key, value in class_map.items():
            labels[int(key)] = value
        labels = torch.LongTensor(labels)

        return features, features_train, labels, idx_train, idx_valid, idx_test

    else:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
        x, y, tx, ty, allx, ally = tuple(objects)

        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            zero_ind = list(set(test_idx_range_full) - set(test_idx_reorder))
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty_extended[zero_ind-min(test_idx_range), np.random.randint(0, y.shape[1], len(zero_ind))] = 1
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = list(range(len(y)))
        idx_valid = list(range(len(y), len(y)+500))

        print('#idx_train', len(idx_train))
        print('#idx_valid', len(idx_valid))
        print('#idx_test', len(idx_test))

        features = preprocess_features(features)
        features = torch.FloatTensor(np.array(features.todense()))
        features_train = torch.clone(features)
        labels = torch.LongTensor(np.where(labels)[1]).unsqueeze(-1)

        return features, features_train, labels, idx_train, idx_valid, idx_test


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def load_csr_mat_from_npz(filename):
    adj = np.load(filename)
    data = adj['data']
    indices = adj['indices']
    indptr = adj['indptr']
    N, M = adj['shape']

    return sp.csr_matrix((data, indices, indptr), (N, M))


def construct_balanced_edge_sets(dataset, sample_type, adj, n_samples):
    indices = adj.indices
    indptr = adj.indptr
    n_nodes = adj.shape[0]

    dic = defaultdict(list)
    for u in range(n_nodes):
        begg, endd = indptr[u: u+2]
        dic[u] = indices[begg: endd]

    edge_set = []
    nonedge_set = []

    # construct edge set
    for u in range(n_nodes):
        for v in dic[u]:
            if v > u:
                edge_set.append((u, v))
    n_samples = len(edge_set)

    # random sample equal number of pairs to compose a nonoedge set
    while 1:
        u = np.random.choice(n_nodes)
        v = np.random.choice(n_nodes)
        if v not in dic[u] and u not in dic[v]:
            nonedge_set.append((u, v))
            if len(nonedge_set) == n_samples: break

    print(f'sampling done! len(edge_set) = {len(edge_set)}, len(nonedge_set) = {len(nonedge_set)}')

    return (edge_set, nonedge_set), list(range(n_nodes))


def construct_edge_sets(dataset, sample_type, adj, n_samples):
    indices = adj.indices
    indptr = adj.indptr
    n_nodes = adj.shape[0]

    # construct edge set
    edge_set = []
    while 1:
        u = np.random.choice(n_nodes)
        begg, endd = indptr[u: u+2]
        v_range = indices[begg: endd]
        if len(v_range):
            v = np.random.choice(v_range)
            edge_set.append((u, v))
            if len(edge_set) == n_samples: break

    # construct non-edge set
    nonedge_set = []

    # # randomly select non-neighbors
    # for _ in tqdm(range(n_samples)):
    #     u = np.random.choice(n_nodes)
    #     begg, endd = indptr[u: u+2]
    #     v_range = indices[begg: endd]
    #     while 1:
    #         v = np.random.choice(n_nodes)
    #         if v not in v_range:
    #             nonedge_set.append((u, v))
    #             break

    # randomly select nodes with two-hop distance
    while 1:
        u = np.random.choice(n_nodes)
        begg, endd = indptr[u: u+2]
        v_range = indices[begg: endd]

        vv_range_all = []
        for v in v_range:
            begg, endd = indptr[v: v+2]
            vv_range = set(indices[begg: endd]) - set(v_range)
            if vv_range:
                vv_range_all.append(vv_range)

        if vv_range_all:
            vv_range = np.random.choice(vv_range_all)
            vv = np.random.choice(list(vv_range))
            nonedge_set.append((u, vv))
            if len(nonedge_set) == n_samples: break

    return edge_set, nonedge_set


def _get_edge_sets_among_nodes(indices, indptr, nodes):
    # construct edge list for each node
    dic = defaultdict(list)

    for u in nodes:
        begg, endd = indptr[u: u+2]
        dic[u] = indices[begg: endd]

    n_nodes = len(nodes)
    edge_set = []
    nonedge_set = []
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            u, v = nodes[i], nodes[j]
            if v in dic[u]:
                edge_set.append((u, v))
            else:
                nonedge_set.append((u, v))

    print('#nodes =', len(nodes))
    print('#edges_set =', len(edge_set))
    print('#nonedge_set =', len(nonedge_set))
    return edge_set, nonedge_set


def _get_degree(n_nodes, indptr):
    deg = np.zeros(n_nodes, dtype=np.int32)
    for i in range(n_nodes):
        deg[i] = indptr[i+1] - indptr[i]

    ind = np.argsort(deg)
    return deg, ind


def construct_edge_sets_from_random_subgraph(dataset, sample_type, adj, n_samples):
    indices = adj.indices
    indptr = adj.indptr
    n_nodes = adj.shape[0]

    if sample_type == 'unbalanced':
        indice_all = range(n_nodes)

    else:
        deg, ind = _get_degree(n_nodes, indptr)
        # unique, count = np.unique(deg, return_counts=True)
        # l = list(zip(unique, count))
        # print(l)
        # print(len(np.where(deg <= 10)[0]))
        # print(len(np.where(deg >= 10)[0]))

        if dataset.startswith('twitch'):
            lo = 5 if 'PTBR' not in dataset else 10
            hi = 10
        elif dataset in ( 'flickr', 'ppi' ) or dataset.startswith('deezer'):
            lo = 15
            hi = 30
        elif dataset in ( 'cora' ):
            lo = 3
            hi = 4
        elif dataset in ( 'citeseer' ):
            lo = 3
            hi = 3
        elif dataset in ( 'pubmed' ):
            lo = 10
            hi = 10
        else:
            raise NotImplementedError(f'lo and hi for dataset = {dataset} not set!')

        if sample_type == 'unbalanced-lo':
            indice_all = np.where(deg <= lo)[0]
        else:
            indice_all = np.where(deg >= hi)[0]

    print('#indice =', len(indice_all))

    nodes = np.random.choice(indice_all, n_samples, replace=False)   # choose from low degree nodes

    return _get_edge_sets_among_nodes(indices, indptr, nodes), nodes



def construct_edge_sets_through_bfs(sample_type, adj, n_hop):
    indices = adj.indices
    indptr = adj.indptr
    n_nodes = adj.shape[0]

    deg, ind = _get_degree(n_nodes, indptr)

    sorted_deg = deg[ind]

    unique, count = np.unique(deg, return_counts=True)
    l = list(zip(unique, count))
    print(l)

    deg_lo = sorted_deg[np.where(sorted_deg)[0][0]]
    deg_hi = sorted_deg[n_nodes - 1]

    print(deg_lo, deg_hi)

    # indice_lo = np.where(deg == deg_lo)[0]
    # indice_hi = np.where(deg == deg_hi)[0]

    indice_lo = np.where(deg <= 5)[0]
    # indice_hi = np.where(deg >= 100)[0]
    # print(len(indice_lo), len(indice_hi))

    # randomly sample a starting node
    # may replace with choosing the node with the highest/lowest degree later
    # src = np.random.choice(n_nodes)
    src = np.random.choice(indice_lo)

    que = Queue()
    vis = np.zeros(n_nodes, dtype=np.int8)

    que.put((src, 0))
    vis[src] = 1

    while 1:
        if que.empty(): break

        head = que.get()
        u, dep = head

        if dep == n_hop: continue

        begg, endd = indptr[u: u+2]
        v_range = indices[begg: endd]
        for v in v_range:
            if vis[v]: continue
            que.put((v, dep+1))
            vis[v] = 1

    nodes = np.where(vis)[0]
    return _get_edge_sets_among_nodes(indices, indptr, nodes)


def load_edges_from_npz(filename):
    adj = np.load(filename)
    indices = adj['indices']
    indptr = adj['indptr']
    n_nodes = adj['shape'][0]
    edges = []
    for i in tqdm(range(n_nodes)):
        begg = indptr[i]
        endd = indptr[i+1]
        edges += [(i, elem) for elem in indices[begg:endd] if elem > i]
    return np.asarray(edges)

def graph_reader(args, dataset="cora", n_nodes=-1):
    if dataset.startswith('twitch') or dataset.startswith('wikipedia/'):
        identifier = dataset[dataset.find('/')+1:]
        data = pd.read_csv('./data/{}/musae_{}_edges.csv'.format(dataset, identifier))
        edges = data.values
        adj = sp.csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(n_nodes, n_nodes),
                            dtype=np.float32)
        return adj + adj.T

    elif dataset.startswith('deezer'):
        identifier = dataset[dataset.find('/')+1:]
        data = pd.read_csv('./data/deezer/{}_edges.csv'.format(identifier))
        edges = data.values
        adj = sp.csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(n_nodes, n_nodes),
                            dtype=np.float32)
        adj += adj.T

        degree = np.zeros(n_nodes, dtype=np.int64)
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1

        features = np.zeros((n_nodes, 500))
        features[np.arange(n_nodes), degree] = 1

        return adj, torch.FloatTensor(features)

    elif dataset == 'facebook':
        data = pd.read_csv('./data/facebook/musae_facebook_edges.csv')
        return data.values

    elif dataset in ('reddit', 'flickr', 'ppi', 'ppi-large'):
        adj_full = load_csr_mat_from_npz(f'./data/{dataset}/adj_full.npz')
        print(f'loading {dataset} graph done!')
        return adj_full

    else:
        with open("data/ind.{}.{}".format(dataset, 'graph'), 'rb') as f:
            if sys.version_info > (3, 0):
                graph = pkl.load(f, encoding='latin1')
            else:
                graph = pkl.load(f)

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
        return sp.csr_matrix(adj_normalized)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))*1.0
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    return mx

def t_normalize(tensor):
    """Row-normalize sparse matrix"""

    rowsum = tensor.sum(1)
    r_inv = rowsum ** -1
    r_inv[torch.isinf(r_inv)] = 0
    r_mat_inv = torch.diag(r_inv)
    ret = torch.mm(r_mat_inv, tensor)

    return ret


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def aug_normalized_adjacency(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def gcn(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return (sp.eye(adj.shape[0]) + d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)).tocoo()


def bingge_norm_adjacency(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return (d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt) +  sp.eye(adj.shape[0])).tocoo()


def normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return (d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)).tocoo()


def random_walk(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv = np.power(row_sum, -1.0).flatten()
   d_mat = sp.diags(d_inv)
   return d_mat.dot(adj).tocoo()


def aug_random_walk(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv = np.power(row_sum, -1.0).flatten()
   d_mat = sp.diags(d_inv)
   return (d_mat.dot(adj)).tocoo()


def fetch_normalization(type):
   switcher = {
       'FirstOrderGCN': gcn,   # A' = I + D^-1/2 * A * D^-1/2
       'BingGeNormAdj': bingge_norm_adjacency, # A' = I + (D + I)^-1/2 * (A + I) * (D + I)^-1/2
       'NormAdj': normalized_adjacency,  # D^-1/2 * A * D^-1/2
       'AugRWalk': aug_random_walk,  # A' = (D + I)^-1*(A + I)
       'RWalk': random_walk,  # A' = D^-1*A
       'AugNormAdj': aug_normalized_adjacency,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
   }
   func = switcher.get(type, lambda: "Invalid normalization technique.")
   return func
