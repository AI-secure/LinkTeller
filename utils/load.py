from collections import OrderedDict
import itertools
import json
import numpy as np
import os
import pandas as pd
import pickle as pkl
import random
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import sys
import torch
from tqdm import tqdm

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def get_noise(noise_type, row, col, seed, eps=10, delta=1e-5):
    np.random.seed(seed)
    size = row * col

    if noise_type == 'laplace':
        noise = np.random.laplace(0, 2/eps, size)
    elif noise_type == 'gaussian':
        c = np.sqrt(2*np.log(1.25/delta))
        stddev = c * np.sqrt(2) / eps
        noise = np.random.normal(0, stddev, size)
    else:
        raise NotImplementedError('noise {} not implemented!'.format(noise_type))

    return noise.reshape(row, col)


def feature_reader(dataset="cora", scale='large', train_ratio=0.5, feature_size=-1):
    if dataset.startswith('twitch/') or dataset.startswith('facebook'):
        if dataset.startswith('twitch'):
            language = dataset[dataset.find('/')+1:]
            filename = './data/{}/musae_{}_features.json'.format(dataset, language)
        else:
            filename = './data/facebook/musae_facebook_features.json'
        with open(filename) as f:
            data = json.load(f)
            items = sorted(set(itertools.chain.from_iterable(data.values())))
            feature_map = {elem: idx for idx, elem in enumerate(items)}
            features = np.zeros((len(data), len(feature_map)))
            for idx, elem in data.items():
                vec = np.zeros(len(feature_map))
                feature = list(map(lambda x: feature_map[x], elem))
                vec[feature] = 1
                features[int(idx), :] = vec

        if dataset.startswith('twitch'):
            data = pd.read_csv('./data/{}/musae_{}_target.csv'.format(dataset, language))
            labels = list(map(int, data['mature'].values))
        else:
            data = pd.read_csv('./data/facebook/musae_facebook_target.csv')
            labels = data['page_type'].values.tolist()
            all_labels = sorted(set(labels))
            label_dict = {label: i for i, label in enumerate(all_labels)}
            labels = list(map(label_dict.get, labels))

        features = normalize(features)
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)

        unique, count = torch.unique(labels, return_counts=True)
        print('count', count)

        len_all = len(labels)
        len_valid = len_test = int(0.1 * len_all)
        len_train = int(train_ratio * len_all)
        idx_all = list(range(len_all))

        random.seed(42)
        random.shuffle(idx_all)
        idx_test = idx_all[:len_test]
        idx_valid = idx_all[len_test:len_test+len_valid]
        idx_train = idx_all[len_test+len_valid:len_test+len_valid+len_train]

        return features, labels, idx_train, idx_valid, idx_test

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

        ''' test: tx (test_idx_range)
            valid: 500 in training (allx)
            train: 1% ~ 10% of training (allx), step = 1%
        '''

        if scale == 'small':
            idx_test = test_idx_range.tolist()
            idx_train = list(range(len(y)))
            idx_valid = list(range(len(y), len(y)+500))

        elif scale == 'large':
            idx_train, idx_valid = train_test_split(list(range(len(labels))), test_size=0.2, random_state=42)
            idx_valid, idx_test = train_test_split(idx_valid, test_size=0.5, random_state=42)

        elif scale == '':
            idx_test = test_idx_range.tolist()

            len_all = len(ally)
            len_train, len_valid = int(train_ratio * len_all), 500
            idx_allx = list(range(len_all))

            random.seed(42)
            random.shuffle(idx_allx)
            idx_valid = idx_allx[:len_valid]
            idx_train = idx_allx[len_valid:len_valid+len_train]

        else:
            raise NotImplementedError('scale {} not implemented!'.format(scale))

        if feature_size != -1:
            np.random.randint(42)
            idx = np.random.randint(0, features.shape[1], feature_size)
            features = features[:, idx]

        print('#idx_train', len(idx_train))
        print('#idx_valid', len(idx_valid))

        features = normalize(features)
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(labels)[1])

        return features, labels, idx_train, idx_valid, idx_test


def graph_reader(dataset="cora"):
    if dataset.startswith('twitch'):
        language = dataset[dataset.find('/')+1:]
        data = pd.read_csv('./data/{}/musae_{}_edges.csv'.format(dataset, language))
        return data.values

    elif dataset == 'facebook':
        data = pd.read_csv('./data/facebook/musae_facebook_edges.csv')
        return data.values

    else:
        with open("data/ind.{}.{}".format(dataset, 'graph'), 'rb') as f:
            if sys.version_info > (3, 0):
                graph = pkl.load(f, encoding='latin1')
            else:
                graph = pkl.load(f)

        edges = []
        for src, neighbor in graph.items():
            edges += [(src, dst) for dst in neighbor if dst > src]
        edges = np.asarray(edges)

        return edges


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))*1.0
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

