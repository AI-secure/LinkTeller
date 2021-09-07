from pandas.core.base import DataError
from sklearn.decomposition import PCA
import logging
import os
import os.path as osp
import numpy as np
import random
import scipy.sparse as sp
from sklearn.metrics import f1_score, average_precision_score
from tensorboardX import SummaryWriter
import time
from tqdm import tqdm, trange

import torch
import torch.nn.functional as F
import torch.optim as optim

from attacker import Attacker
from gcn import GCN, GCN3, ProjectionGCN, MLP, SGC, DeGCN
from utils import EarlyStopping, get_noise

class GCNTrainer():
    def __init__(self, args, subdir='', worker=None):
        self.args = args

        self.worker = worker
        self.loss_func = F.cross_entropy if self.worker.multi_label == 1 \
            else F.binary_cross_entropy_with_logits
        self.mode = self.worker.mode
        self.dataset = self.worker.dataset
        self.subdir = subdir

        self.gcnt_train = self.gcnt_valid = 0
        if self.args.early:
            self.early_stopping = EarlyStopping(patience=self.args.patience)

        if subdir:
            self.init_all_logging(subdir)


    def calc_loss(self, input, target):
        if self.loss_func == F.cross_entropy:
            return self.loss_func(input, target.squeeze())
        else:
            return self.loss_func(input, target.float())


    def init_all_logging(self, subdir):
        tflog_path = os.path.join('tflogs_{}'.format(self.dataset), subdir)
        self.model_path = os.path.join('model_{}'.format(self.dataset), subdir)
        self.writer = SummaryWriter(log_dir=tflog_path)
        if not os.path.exists(self.model_path): os.makedirs(self.model_path)


    def init_model(self, model_path=''):
        # Model and optimizer
        if self.mode in ( 'sgc-clean', 'sgc' ):
            self.model = SGC(nfeat=self.worker.n_features,
                        nclass=self.worker.n_classes)

        elif self.mode in ( 'degree_mlp', 'basic_mlp' ):
            self.model = MLP(nfeat=self.worker.n_features,
                        nhid=self.args.hidden,
                        nclass=self.worker.n_classes,
                        dropout=self.args.dropout, 
                        size=self.worker.n_nodes,
                        args=self.args)

        elif self.mode in ( 'degcn-clean' ):
            self.model = DeGCN(nfeat=self.worker.n_features,
                        nhid=self.args.hidden,
                        nclass=self.worker.n_classes,
                        dropout=self.args.dropout)

        elif self.mode in ( 'vanilla-clean', 'vanilla' ) or not self.args.fnormalize:
            if self.args.n_layer == 2:
                self.model = GCN(nfeat=self.worker.n_features,
                            nhid=self.args.hidden,
                            nclass=self.worker.n_classes,
                            dropout=self.args.dropout)
            elif self.args.n_layer == 3:
                self.model = GCN3(nfeat=self.worker.n_features,
                            nhid1=self.args.hidden1,
                            nhid2=self.args.hidden2,
                            nclass=self.worker.n_classes,
                            dropout=self.args.dropout)
            else:
                raise NotImplementedError(f'n_layer = {self.args.n_layer} not implemented!')

        elif self.mode in ( 'clusteradj-clean', 'clusteradj' ):
            self.model = ProjectionGCN(nfeat=self.worker.n_features,
                        nhid=self.args.hidden,
                        nclass=self.worker.n_classes,
                        dropout=self.args.dropout, 
                        projection=self.worker.prj,
                        size=self.worker.n_nodes,
                        args=self.args)

        else:
            raise NotImplementedError('mode = {} no corrsponding model!'.format(self.mode))

        if model_path:
            self.model.load_state_dict(torch.load(model_path))
            print('load model from {} done!'.format(model_path))
            self.model_path = model_path
        else:
            self.optimizer = optim.Adam(self.model.parameters(),
                                lr=self.args.lr, weight_decay=self.args.weight_decay)
        if torch.cuda.is_available():
            self.model.cuda()


    def forward(self, mode='train'):
        if self.mode in ( 'degcn-clean' ):
            output = self.model(self.worker.features, self.worker.adj, self.worker.sub_adj)

        elif self.mode in ( 'sgc-clean' ):
            if self.dataset in ('reddit', 'flickr', 'ppi', 'ppi-large', 'cora', 'citeseer', 'pubmed') and mode == "train":
                output = self.model(self.worker.features_train)

            elif self.worker.transfer:
                output = self.model(self.worker.features_1) if mode == 'train' \
                    else self.model(self.worker.features_2)

            else:
                output = self.model(self.worker.features)

        else:
            if self.dataset in ('reddit', 'flickr', 'ppi', 'ppi-large', 'cora', 'citeseer', 'pubmed') \
                or self.dataset.startswith('twitch-train'):
                output = self.model(self.worker.features_train, self.worker.adj_train) if mode == 'train' \
                    else self.model(self.worker.features, self.worker.adj_full)

            elif self.worker.transfer:
                output = self.model(self.worker.features_1, self.worker.adj_1) if mode == 'train' \
                    else self.model(self.worker.features_2, self.worker.adj_2)

            else:
                output = self.model(self.worker.features, self.worker.adj)

        return output


    def train_one_epoch(self, epoch):

        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()

        output = self.forward(mode='train')
        output = output[self.worker.idx_train] if self.dataset in ( 'cora', 'citeseer', 'pubmed' ) else output

        target_labels = self.worker.labels_1 if self.worker.transfer \
                    else self.worker.labels[self.worker.idx_train]

        loss_train = self.calc_loss(output, target_labels)
        acc_train = self.f1_score(output, target_labels)

        loss_train.backward()

        self.optimizer.step()

        self.writer.add_scalar('train/loss', loss_train, self.gcnt_train)
        self.writer.add_scalar('train/acc', acc_train[0], self.gcnt_train)
        self.gcnt_train += 1

        if self.worker.transfer:   # no validation set
            self.model.eval()
            output = self.forward(mode='valid')
            loss_val = self.calc_loss(output, self.worker.labels_2)
            acc_val = self.f1_score(output, self.worker.labels_2)
            self.writer.add_scalar('valid/loss', loss_val, self.gcnt_valid)
            self.writer.add_scalar('valid/acc', acc_val[0], self.gcnt_valid)
            self.gcnt_valid += 1

            output_info = 'Epoch: {:04d}'.format(epoch+1),\
                'loss_train: {:.4f}'.format(loss_train.item()),\
                'acc_train: {:.4f}'.format(acc_train[0].item()),\
                'time: {:.4f}s'.format(time.time() - t)
            logging.info(output_info)
            return loss_train

        if not self.args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            self.model.eval()
            output = self.forward(mode='valid')

        loss_val = self.calc_loss(output[self.worker.idx_val], self.worker.labels[self.worker.idx_val])
        acc_val = self.f1_score(output[self.worker.idx_val], self.worker.labels[self.worker.idx_val])
        self.writer.add_scalar('valid/loss', loss_val, self.gcnt_valid)
        self.writer.add_scalar('valid/acc', acc_val[0], self.gcnt_valid)
        self.gcnt_valid += 1

        if self.args.early:
            self.early_stopping(loss_val, self.model)

        output_info = 'Epoch: {:04d}'.format(epoch+1),\
            'loss_train: {:.4f}'.format(loss_train.item()),\
            'acc_train: {:.4f}'.format(acc_train[0].item()),\
            'loss_val: {:.4f}'.format(loss_val.item()),\
            'acc_val: {:.4f}'.format(acc_val[0].item()),\
            'time: {:.4f}s'.format(time.time() - t)

        logging.info(output_info)
        return loss_train


    def train(self):
        # Train model
        t_total = time.time()

        if self.args.display:
            epochs = trange(self.args.num_epochs, desc='Progress')
        else:
            epochs = range(self.args.num_epochs)

        # if self.mode in ( 'clusteradj-clean', 'clusteradj' ):
        #     data = {
        #         # 'adj': self.worker.adj
        #         'values': self.worker.adj.coalesce().values(),
        #         'indices': self.worker.adj.coalesce().indices(),
        #     }
        # else:
        #     data = {'adj': self.worker.adj}

        # torch.save(data, 'temp_adj.pt')

        for epoch in epochs:
            logging.info('[epoch {}]'.format(epoch))
            output = self.train_one_epoch(epoch)
            if self.args.display:
                epochs.set_description(f"Train Loss: {output}")

            if self.args.early and self.early_stopping.early_stop:
                self.model = self.early_stopping.best_model
                logging.info(f'early stop at epoch {epoch}')
                break

        torch.save(self.model.state_dict(), os.path.join(self.model_path, 'model.pt'))

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


    def f1_score(self, output, labels):
        if self.worker.multi_label == 1:
            preds = F.softmax(output, dim=1)
            preds = preds.max(1)[1].type_as(labels)
            return f1_score(labels.cpu(), preds.detach().cpu(), average='micro'), \
                f1_score(labels.cpu(), preds.detach().cpu(), average='macro'), \
                f1_score(labels.cpu(), preds.detach().cpu(), average='weighted')
            # unique, count = torch.unique(preds, return_counts=True)
            # correct = preds.eq(labels).double()
            # correct = correct.sum()
            # return correct / len(labels)

        else:   # multi_label
            preds = torch.sigmoid(output) > 0.5
            return f1_score(labels.cpu(), preds.detach().cpu(), average='micro'), \
                f1_score(labels.cpu(), preds.detach().cpu(), average='macro'), \
                f1_score(labels.cpu(), preds.detach().cpu(), average='weighted')


    def rare_class_f1(self, output, labels):
        # identify the rare class
        ind = [torch.where(labels==0)[0],
                torch.where(labels==1)[0]]
        rare_class = int(len(ind[0]) > len(ind[1]))

        preds = F.softmax(output, dim=1).max(1)

        ap_score = average_precision_score(labels.cpu() if rare_class==1 else 1-labels.cpu(), preds[0].detach().cpu())

        preds = preds[1].type_as(labels)

        TP = torch.sum(preds[ind[rare_class]] == rare_class).item()
        T = len(ind[rare_class])
        P = torch.sum(preds == rare_class).item()

        if P == 0: return 0

        precision = TP / P
        recall = TP / T
        F1 = 2 * (precision * recall) / (precision + recall)
        return F1, precision, recall, ap_score


    def eval_degree(self, output):
        degrees = self.worker.calculate_degree()
        if self.dataset.startswith('twitch'):
            path = self.dataset.replace('/', '_')
        else:
            path = self.dataset
        torch.save(degrees, f'{path}_degrees.pt')

        # unique = np.unique(degrees)
        # acc_list = np.zeros_like(degrees)
        # total_list = np.zeros_like(degrees)

        # idx_list = list(range(self.worker.n_nodes_2)) if self.dataset.startswith( 'twitch' ) else self.worker.idx_test
        # labels = self.worker.labels_2 if self.dataset.startswith( 'twitch' ) else self.worker.labels


        # for i, value in enumerate(unique):
        #     indice_cur = np.intersect1d(np.where(degrees == value)[0], idx_list, assume_unique=True)
        #     if indice_cur.size == 0: continue
        #     acc_cur = self.f1_score(output[indice_cur], labels[indice_cur])

        #     acc_list[i] = acc_cur[0]
        #     total_list[i] = len(indice_cur)

        # degree_info = 'acc for different node degree: {}'.format(list(zip(unique, acc_list)))
        # # torch.save(list(zip(unique, acc_list)), 'degree_{}_{}.pt'.format(mode, self.subdir))
        # # torch.save(list(zip(unique, total_list), 'total_num.pt'))
        # print(degree_info)
        # logging.info(degree_info)


    def eval_output(self, output, mode='clean', eval_degree=False):
        if self.args.attack:
            self.attacker = Attacker(args=self.args, model=self.model, worker=self.worker)
            self.attacker.prepare_test_data()

            t = time.time()
            if self.args.attack_mode == 'efficient':
                if self.args.sample_type == 'balanced-full':
                    self.attacker.link_prediction_attack_efficient_balanced()
                else:
                    self.attacker.link_prediction_attack_efficient()
            elif self.args.attack_mode == 'naive':
                self.attacker.link_prediction_attack()
            elif self.args.attack_mode in ( 'baseline', 'baseline-feat' ):
                if self.args.sample_type == 'balanced-full':
                    self.attacker.baseline_attack_balanced()
                else:
                    self.attacker.baseline_attack()
            # self.attacker.link_prediction_attack()
            print(f'attacks done using {time.time() - t} seconds!')

        if not self.worker.transfer:
            loss_valid = self.calc_loss(output[self.worker.idx_val], self.worker.labels[self.worker.idx_val])
            acc_valid = self.f1_score(output[self.worker.idx_val], self.worker.labels[self.worker.idx_val])

            # result on validation set
            output_info = f'''[{mode}] Validation set results: '''\
                        f'''loss = {loss_valid.item():.4f} '''\
                        f'''f1_score = {acc_valid[0].item():.4f}'''
            print(output_info)
            logging.info(output_info)

        if self.dataset.startswith('twitch-train'): return

        output_labels = output if self.worker.transfer \
                    else output[self.worker.idx_test]
        target_labels = self.worker.labels_2 if self.worker.transfer \
                    else self.worker.labels[self.worker.idx_test]

        loss_test = self.calc_loss(output_labels, target_labels)
        acc_test = self.f1_score(output_labels, target_labels) if not self.worker.transfer \
                    else self.rare_class_f1(output_labels, target_labels)

        # if 'model.pt' in self.model_path:
        #     labels_path = self.model_path.replace('model.pt', f'labels.pt')
        # else:
        #     labels_path = osp.join(self.model_path, 'labels.pt')
        # torch.save({'output': output_labels.cpu(),
        #             'target': target_labels.cpu(),
        #     }, labels_path)
        # print(f'labels saved to {labels_path}!')
        # logging.info(f'labels saved to {labels_path}!')

        # a0 = self.worker.adj.cpu().to_dense().numpy()[633,:]
        # print(np.where(a0!=0))

        if eval_degree:
            self.eval_degree(output)

        output_info = f'''[{mode}] Test set results: '''\
                    f'''loss = {loss_test.item():.4f} '''
        output_info += f'rare_class_f1 = {acc_test[0]:.4f} prec = {acc_test[1]:.4f} reca = {acc_test[2]:.4f} ap_score = {acc_test[3]:.4f}' if self.worker.transfer else \
                f'''f1_score [micro, macro, weighted] = {acc_test[0].item():.4f} {acc_test[1].item():.4f} {acc_test[2].item():.4f}''' 
        print(output_info)
        logging.info(output_info)


    def test(self, eval_degree=False):
        self.model.eval()

        # if self.mode in ( 'vanilla', 'clusteradj', 'degcn' ):
        #     if self.mode == 'clusteradj' and self.args.fnormalize:
        #         logging.info(f'eventual coeff: {self.model.gc1.coeff.item()}, {self.model.gc2.coeff.item()}')

        #     # test on noisy graph
        #     output = self.forward(mode='test')
        #     self.eval_output(output, 'noisy', eval_degree)

        #     # test on clean graph
        #     self.worker.update_adj()
        #     output = self.forward(mode='test')
        #     self.eval_output(output, 'clean', eval_degree)

        # else:
        # test on clean graph
        output = self.forward(mode='test')
        self.eval_output(output, 'clean', eval_degree=eval_degree)


    def __del__(self):
        if hasattr(self, 'writer'):
            self.writer.close()