import copy
import logging
import numpy as np
import os
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, average_precision_score
from tensorboardX import SummaryWriter
import time
from tqdm import trange, tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from gcn import GCN, ProjectionGCN, MLP, SGC, SingleHiddenLayerMLP, OneLayerMLP
from utils import EarlyStopping, get_noise

class MLPTrainer():
    def __init__(self, args, subdir='', worker=None, train=True):
        self.args = args

        self.worker = worker
        self.loss_func = F.cross_entropy if self.worker.multi_label == 1 \
            else F.binary_cross_entropy_with_logits
        self.mode = self.worker.mode
        self.dataset = self.worker.dataset
        self.subdir = subdir
        self.is_train = train

        self.gcnt_train = self.gcnt_valid = 0
        if self.args.early:
            self.early_stopping = EarlyStopping(patience=self.args.patience)

        if subdir:
            self.init_all_logging(subdir)

        self.transfer = (not self.dataset.startswith('twitch-train') and self.dataset.startswith('twitch')) or \
                        self.dataset.startswith('wikipedia') or \
                        self.dataset.startswith('deezer')            

        self.prepare_data()


    def calc_loss(self, input, target):
        if self.loss_func == F.cross_entropy:
            return self.loss_func(input, target.squeeze())
        else:
            return self.loss_func(input, target.float())


    def prepare_data(self):
        if self.is_train:
            if self.transfer:
                self.train_loader = DataLoader(TensorDataset(
                    self.worker.features_1,
                    self.worker.labels_1
                ), batch_size=self.args.batch_size, shuffle=True)

            else:
                self.train_loader = DataLoader(TensorDataset(
                    self.worker.features[self.worker.idx_train],
                    self.worker.labels[self.worker.idx_train]
                ), batch_size=self.args.batch_size, shuffle=True)


    def init_all_logging(self, subdir):
        tflog_path = os.path.join('tflogs_{}'.format(self.dataset), subdir)
        self.model_path = os.path.join('model_{}'.format(self.dataset), subdir)
        self.writer = SummaryWriter(log_dir=tflog_path)
        if not os.path.exists(self.model_path): os.makedirs(self.model_path)


    def init_model(self, model_path=''):
        # Model and optimizer
        if self.mode in ( 'mlp' ):
            if self.args.depth == 1:
                self.model = OneLayerMLP(nfeat=self.worker.n_features,
                                        nclass=self.worker.n_classes)
            elif self.args.depth == 2:
                self.model = SingleHiddenLayerMLP(nfeat=self.worker.n_features,
                            nhid=self.args.hidden,
                            nclass=self.worker.n_classes,
                            dropout=self.args.dropout)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))
            print('load model from {} done!'.format(model_path))
        else:
            self.optimizer = optim.Adam(self.model.parameters(),
                                lr=self.args.lr, weight_decay=self.args.weight_decay)
        if torch.cuda.is_available():
            self.model.cuda()


    def train_one_epoch(self, epoch):
        # training
        self.model.train()

        loss_seq = []
        acc_seq = []

        if self.args.batch:
            for features, labels in self.train_loader:
                output = self.model(features)

                loss = self.calc_loss(output, labels)
                acc = self.f1_score(output, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_seq.append(loss.item())
                acc_seq.append(acc[0].item())
                self.gcnt_train += 1
                if self.gcnt_train % 10 == 0:
                    self.writer.add_scalar('train/loss', np.mean(loss_seq), self.gcnt_train)
                    self.writer.add_scalar('train/acc', np.mean(acc_seq), self.gcnt_train)
                    loss_seq = []
                    acc_seq = []


            if not self.transfer:
                # validation
                output = self.model(self.worker.features[self.worker.idx_val])
                loss_val = self.calc_loss(output, self.worker.labels[self.worker.idx_val])
                acc_val = self.f1_score(output, self.worker.labels[self.worker.idx_val])
                self.gcnt_valid += 1
                self.writer.add_scalar('valid/loss', loss_val, self.gcnt_valid)
                self.writer.add_scalar('valid/acc', acc_val[0], self.gcnt_valid)

        else:
            output = self.model(self.worker.features_1)
            loss = self.calc_loss(output, self.worker.labels_1)
            acc = self.f1_score(output, self.worker.labels_1)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def train(self):
        # Train model
        t_total = time.time()

        for epoch in tqdm(range(self.args.num_epochs)):
            logging.info('[epoch {}]'.format(epoch))
            self.train_one_epoch(epoch)

            if self.args.early and self.early_stopping.early_stop:
                self.model = self.early_stopping.best_model
                logging.info(f'early stop at epoch {epoch}')
                break

        torch.save(self.model.state_dict(), os.path.join(self.model_path, 'model.pt'))

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


    def eval_output(self):
        if not self.transfer:
            output = self.model(self.worker.features[self.worker.idx_val])
            loss_val = self.calc_loss(output, self.worker.labels[self.worker.idx_val])
            acc_val = self.f1_score(output, self.worker.labels[self.worker.idx_val])

            output_info = f'''Valid set results: '''\
                f'''loss = {loss_val.item():.4f} '''\
                f'''f1_score = {acc_val[0].item():.4f} '''
            print(output_info)
            logging.info(output_info)

        output = self.model(self.worker.features_2) if self.transfer \
            else self.model(self.worker.features[self.worker.idx_test])
        target = self.worker.labels_2 if self.transfer \
            else self.worker.labels[self.worker.idx_test]
        loss_test = self.calc_loss(output, target)
        acc_test = self.f1_score(output, target) if not self.worker.transfer \
                    else self.rare_class_f1(output, target)

        output_info = f'''Test set results: '''\
            f'''loss = {loss_test.item():.4f} '''
        output_info += f'rare_class_f1 = {acc_test[0]:.4f} prec = {acc_test[1]:.4f} reca = {acc_test[2]:.4f} ap_score = {acc_test[3]:.4f}' if self.worker.transfer else \
                f'''f1_score [micro, macro, weighted] = {acc_test[0].item():.4f} {acc_test[1].item():.4f} {acc_test[2].item():.4f}''' 
        print(output_info)
        logging.info(output_info)


    def test(self, eval_degree=False):
        self.model.eval()

        self.eval_output()


    def __del__(self):
        if hasattr(self, 'writer'):
            self.writer.close()