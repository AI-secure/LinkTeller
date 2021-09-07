import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .layers import GraphConvolution, ProjectionGraphConvolution, DegreeMLP, BasicMLP

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        # self.gc = GraphConvolution(nfeat, nclass)
        self.dropout = dropout

        print('init GCN model done!')

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # x = self.gc(x, adj)
        return x
        # return F.log_softmax(x, dim=1)


class GCN3(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nclass, dropout):
        super(GCN3, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2)
        self.gc3 = GraphConvolution(nhid2, nclass)
        # self.gc = GraphConvolution(nfeat, nclass)
        self.dropout = dropout

        print('init GCN model done!')

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        return x


class DeGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(DeGCN, self).__init__()

        self.gc1_1 = GraphConvolution(nfeat, nhid)
        self.gc1_2 = GraphConvolution(nfeat, nhid)
        self.gc1_3 = GraphConvolution(nfeat, nhid)
        self.gc1 = [self.gc1_1, self.gc1_2, self.gc1_3]

        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

        print('init DeGCN model done!')

    def forward(self, x, adj, sub_adj):
        x_d = F.dropout(x, self.dropout, training=self.training)
        x = self.gc1[0](x_d, sub_adj[0])
        for i in range(1, len(self.gc1)):
            x += self.gc1[i](x_d, sub_adj[i])
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class ProjectionGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, projection, size, args):
        super(ProjectionGCN, self).__init__()

        self.gc1 = ProjectionGraphConvolution(nfeat, nhid, projection, size, trainable=args.trainable)
        self.gc2 = ProjectionGraphConvolution(nhid, nclass, projection, size, trainable=args.trainable)
        self.dropout = dropout

        print(f'init ProjectionGCN model done!')

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class SingleHiddenLayerMLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(SingleHiddenLayerMLP, self).__init__()

        self.W1 = nn.Linear(nfeat, nhid)
        self.W2 = nn.Linear(nhid, nclass)
        self.dropout = dropout

        self.W = nn.Linear(nfeat, nclass)

        print(f'init SingleHiddeenLayerMLP model done!')

    def forward(self, x):
        x = F.relu(self.W1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.W2(x)
        return x


class OneLayerMLP(nn.Module):
    def __init__(self, nfeat, nclass):
        super(OneLayerMLP, self).__init__()

        self.W = nn.Linear(nfeat, nclass)

        print(f'init OneLayerMLP model done!')

    def forward(self, x):
        x = self.W(x)
        return x


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, size, args):
        super(MLP, self).__init__()

        layer = {
            'degree_mlp': DegreeMLP,
            'basic_mlp': BasicMLP,
        }.get(args.mode)

        self.gc1 = layer(nfeat, nhid, size, args=args)
        self.gc2 = layer(nhid, nclass, size, args=args)
        self.dropout = dropout

        print(f'init MLP model of layer_type = {layer.__name__} done!')

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x):
        return self.W(x)


class BinaryLR(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat):
        super(BinaryLR, self).__init__()

        self.w = Parameter(torch.FloatTensor(nfeat, 1))
        self.b = Parameter(torch.FloatTensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1
        self.w.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return torch.mm(x, self.w) + self.b
