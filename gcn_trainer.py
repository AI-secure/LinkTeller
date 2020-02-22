from sklearn.decomposition import PCA
import logging
import os
import numpy as np
import scipy.sparse as sp
from tensorboardX import SummaryWriter
import time
from tqdm import trange

import torch
import torch.nn.functional as F
import torch.optim as optim

from gcn import GCN, ProjectionGCN

class GCNTrainer():
    def __init__(self, args, subdir='', worker=None):
        self.args = args

        self.worker = worker
        self.mode = self.worker.mode
        self.dataset = self.worker.dataset
        self.subdir = subdir

        self.gcnt_train = self.gcnt_valid = 0
        self.optimal = { 'acc' : 0, 'cnt' : 0 }

        if subdir:
            self.init_all_logging(subdir)


    def init_all_logging(self, subdir):
        tflog_path = os.path.join('tflogs_{}'.format(self.dataset), subdir)
        self.model_path = os.path.join('model_{}'.format(self.dataset), subdir)
        self.writer = SummaryWriter(log_dir=tflog_path)
        if not os.path.exists(self.model_path): os.makedirs(self.model_path)


    def init_model(self, model_path=''):
        # Model and optimizer
        if self.mode in ( 'vanilla-clean', 'vanilla' ):
            self.model = GCN(nfeat=self.worker.n_features,
                        nhid=self.args.hidden,
                        nclass=self.worker.n_classes,
                        dropout=self.args.dropout)

        elif self.mode in ( 'clusteradj-clean', 'clusteradj' ):
            self.model = ProjectionGCN(nfeat=self.worker.n_features,
                        nhid=self.args.hidden,
                        nclass=self.worker.n_classes,
                        dropout=self.args.dropout, 
                        projection=self.worker.prj,
                        size=self.worker.n_nodes)

        else:
            raise NotImplementedError('mode = {} no corrsponding model!'.format(self.mode))

        if model_path:
            self.model.load_state_dict(torch.load(model_path))
            print('load model from {} done!'.format(model_path))
        else:
            self.optimizer = optim.Adam(self.model.parameters(),
                                lr=self.args.lr, weight_decay=self.args.weight_decay)
        if torch.cuda.is_available():
            self.model.cuda()


    def train_one_epoch(self, epoch):

        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(self.worker.features, self.worker.adj)

        loss_train = F.nll_loss(output[self.worker.idx_train], self.worker.labels[self.worker.idx_train])
        acc_train = self.accuracy(output[self.worker.idx_train], self.worker.labels[self.worker.idx_train])
        loss_train.backward()
        self.optimizer.step()

        self.writer.add_scalar('train/loss', loss_train, self.gcnt_train)
        self.writer.add_scalar('train/acc', acc_train, self.gcnt_train)
        self.gcnt_train += 1

        if not self.args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            self.model.eval()
            output = self.model(self.worker.features, self.worker.adj)

        loss_val = F.nll_loss(output[self.worker.idx_val], self.worker.labels[self.worker.idx_val])
        acc_val = self.accuracy(output[self.worker.idx_val], self.worker.labels[self.worker.idx_val])
        self.writer.add_scalar('valid/loss', loss_val, self.gcnt_valid)
        self.writer.add_scalar('valid/acc', acc_val, self.gcnt_valid)
        self.gcnt_valid += 1

        if acc_val > self.optimal['acc']:
            self.optimal['acc'] = acc_val
            self.optimal['cnt'] = 0
        else:
            self.optimal['cnt'] += 1

        output_info = 'Epoch: {:04d}'.format(epoch+1),\
            'loss_train: {:.4f}'.format(loss_train.item()),\
            'acc_train: {:.4f}'.format(acc_train.item()),\
            'loss_val: {:.4f}'.format(loss_val.item()),\
            'acc_val: {:.4f}'.format(acc_val.item()),\
            'time: {:.4f}s'.format(time.time() - t)

        logging.info(output_info)
        return acc_val


    def train(self):
        # Train model
        t_total = time.time()

        if self.args.display:
            epochs = trange(self.args.num_epochs, desc='Valid Acc')
        else:
            epochs = range(self.args.num_epochs)

        for epoch in epochs:
            logging.info('[epoch {}]'.format(epoch))
            acc_val = self.train_one_epoch(epoch)
            if self.args.display:
                epochs.set_description("Valid Acc: %g" % acc_val)
        torch.save(self.model.state_dict(), os.path.join(self.model_path, 'model.pt'))

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


    def accuracy(self, output, labels):
        preds = output.max(1)[1].type_as(labels)
        unique, count = torch.unique(preds, return_counts=True)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)


    def eval_output(self, output, mode='clean', eval_degree=False):
        loss_test = F.nll_loss(output[self.worker.idx_test], self.worker.labels[self.worker.idx_test])
        acc_test = self.accuracy(output[self.worker.idx_test], self.worker.labels[self.worker.idx_test])

        if eval_degree:
            degrees = self.worker.calculate_degree()
            unique = np.unique(degrees)
            acc_list = np.zeros_like(degrees)
            total_list = np.zeros_like(degrees)
            for i, value in enumerate(unique):
                indice_cur = np.intersect1d(np.where(degrees == value)[0], self.worker.idx_test, assume_unique=True)
                if indice_cur.size == 0: continue
                acc_cur = self.accuracy(output[indice_cur], self.worker.labels[indice_cur])
                acc_list[i] = acc_cur
                total_list[i] = len(indice_cur)
            degree_info = 'acc for different node degree: {}'.format(list(zip(unique, acc_list)))
            # torch.save(list(zip(unique, acc_list)), 'degree_{}_{}.pt'.format(mode, self.subdir))
            # torch.save(list(zip(unique, total_list), 'total_num.pt'))
            print(degree_info)
            logging.info(degree_info)

        output_info = "[{}] Test set results:".format(mode),\
            "loss= {:.4f}".format(loss_test.item()),\
            "accuracy= {:.4f}".format(acc_test.item())
        print(output_info)
        logging.info(output_info)


    def test(self, eval_degree=False):
        self.model.eval()

        if self.mode in ( 'vanilla', 'clusteradj' ):
            # test on noisy graph
            output = self.model(self.worker.features, self.worker.adj)
            self.eval_output(output, 'noisy', eval_degree)

            # test on clean graph
            self.worker.update_adj()
            output = self.model(self.worker.features, self.worker.adj)
            self.eval_output(output, 'clean', eval_degree)

        else:
            # test on clean graph
            output = self.model(self.worker.features, self.worker.adj)
            self.eval_output(output, 'clean')


    def __del__(self):
        if hasattr(self, 'writer'):
            self.writer.close()