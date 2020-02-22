import logging
import numpy as np
import operator
import random
from sklearn.cluster import KMeans
import torch
from tqdm import tqdm

class Labeler():
    def __init__(self, features, labels, n_classes, idx_train, idx_val, idx_test):
        self.features = features
        self.labels = labels
        self.n_instances = len(self.labels)
        self.n_classes = n_classes
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test

    def get_knn_labels(self, knn):
        fake_labels = self.labels.clone()

        new_features = self.features.cpu().numpy()

        # 1. prepare training and test data
        train_X, train_y = new_features[self.idx_train], self.labels[self.idx_train]
        val_X, val_y = new_features[self.idx_val], self.labels[self.idx_val]

        if knn == -1:
            # 2. KNN -- selecting the optimal K based on validation set
            upper = len(self.idx_train) // 10
            K_list = [i*10 for i in range(1, upper+1)]
            cnt_same = {K: 0 for K in K_list}
            for i, (X, y) in tqdm(enumerate(zip(val_X, val_y))):
                diff = train_X - X
                dist = np.einsum('ij, ij->i', diff, diff)
                indice = np.argsort(dist)

                for K in K_list:
                    selected_labels = train_y[indice[:K]]

                    unique, count = np.unique(selected_labels, return_counts=True)

                    assert(fake_labels[self.idx_val[i]] == y)
                    label_list = unique[np.argwhere(count == np.amax(count)).flatten().tolist()].tolist()

                    if y in label_list:
                        cnt_same[K] += 1

            cnt_same = sorted(cnt_same.items(), key=operator.itemgetter(1))
            logging.info('cnt_same = {}'.format(cnt_same))
            best_K = cnt_same[-1][0]
        else:
            best_K = knn

        logging.info('best_k = {}'.format(best_K))

        # 3. KNN -- assign labels to validation nodes and testing nodes
        idx_test = list(set(range(self.n_instances)) - set(self.idx_train))
        test_X, test_y = new_features[idx_test], self.labels[idx_test]

        cnt_same_tot = 0
        for i, (X, y) in tqdm(enumerate(zip(test_X, test_y))):
            diff = train_X - X
            dist = np.einsum('ij, ij->i', diff, diff)
            indice = np.argsort(dist)

            selected_labels = train_y[indice[:best_K]]

            unique, count = np.unique(selected_labels, return_counts=True)

            assert(fake_labels[idx_test[i]] == y)
            label_list = unique[np.argwhere(count == np.amax(count)).flatten().tolist()].tolist()
            fake_labels[idx_test[i]] = random.sample(label_list, 1)[0]
            if y == fake_labels[idx_test[i]]:
                cnt_same_tot += 1

        logging.info('cnt_same / cnt_all = {} / {}'.format(cnt_same_tot, len(test_X)))
        return fake_labels


    def get_majority_labels(self, edges, assign_seed):
        fake_labels = self.labels.clone()

        np.random.seed(assign_seed)
        idx_unknown = list(set(range(self.n_instances)) - set(self.idx_train))
        neighbor = {idx : [0 for i in range(self.n_classes)] for idx in idx_unknown}
        for dst, src in edges.tolist():
            if dst in idx_unknown and src in idx_unknown:
                neighbor[src][np.random.randint(0, self.n_classes)] += 1
                neighbor[dst][np.random.randint(0, self.n_classes)] += 1
            elif dst in idx_unknown:
                neighbor[dst][self.labels[src]] += 1
            elif src in idx_unknown:
                neighbor[src][self.labels[dst]] += 1

        cnt_same = 0
        for idx, l in neighbor.items():
            fake_labels[idx] = int(np.argmax(l))
            if self.labels[idx] == fake_labels[idx]:
                cnt_same += 1

        logging.info('cnt_same / cnt_all = {} / {}'.format(cnt_same, len(neighbor)))
        return fake_labels


    def get_naive_labels(self, assign_seed):
        fake_labels = self.labels.clone()
        np.random.seed(assign_seed)
        idx_test = list(set(range(self.n_instances)) - set(self.idx_train))
        fake_labels[idx_test] = np.random.randint(0, self.n_classes, len(idx_test))
        return fake_labels


    def get_cluster_labels(self, features, n_clusters, quota=[], start=0, same_size=False):
        estimator = KMeans(init=f'k-means++', n_clusters=n_clusters, n_init=10)
        km = estimator.fit(features)

        if not same_size:
            return torch.LongTensor(km.labels_)

        else:
            d = km.transform(features)
            indice = np.zeros_like(d.T, dtype=int)
            for i, col in enumerate(d.T):
                indice[i,:] = np.argsort(col)

            labels = [-1] * len(features)
            ptr = [-1] * n_clusters
            nums = [0] * n_clusters
            while 1:
                flag = False
                for i, (a, b) in enumerate(zip(nums, quota)):
                    if a < b:
                        flag = True
                        break

                if not flag:
                    break

                for i in range(n_clusters):
                    if nums[i] == quota[i]: continue
                    while 1:
                        ptr[i] += 1
                        if labels[indice[i][ptr[i]]] == -1:
                            labels[indice[i][ptr[i]]] = i
                            nums[i] +=  1
                            break

            labels = np.asarray(labels) + np.repeat(start, len(labels))

            return torch.LongTensor(labels)


    def get_kmeans_labels(self, n_clusters=10, knn=-1, cluster_method='kmeans', same_size=False):
        fake_labels = self.get_knn_labels(knn)

        labels_vec = np.zeros((self.n_instances, self.n_classes))
        labels_vec[range(self.n_instances), fake_labels] = 1

        features = np.hstack((self.features.numpy(), labels_vec))

        if same_size:
            size = self.n_instances // n_clusters
            rem = self.n_instances % n_clusters
            quota = [size+1] * rem + [size] * (n_clusters-rem)
            logging.info(f'quota = {quota}')
            return self.get_cluster_labels(features, n_clusters, quota=quota, same_size=True)

        else:
            return self.get_cluster_labels(features, n_clusters)


    def get_random_labels(self, n_clusters=10, seed=42):
        np.random.seed(seed)
        return torch.LongTensor(np.random.randint(0, n_clusters, self.n_instances))


    def get_equal_size(self, total, size):
        total, size = int(total), int(size)
        rem = total % size
        num = total // size
        if rem == 0:
            return num, [size] * num

        v1 = (rem - 1) // num + 1
        v2 = (size - rem) // (num + 1)

        if v1 > v2:
            num += 1

        size = total // num
        rem = total % num

        return num, [size+1] * rem + [size] * (num-rem)