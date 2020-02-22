import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution, ProjectionGraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

        print('init GCN model done!')

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class ProjectionGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, projection, size):
        super(ProjectionGCN, self).__init__()

        self.gc1 = ProjectionGraphConvolution(nfeat, nhid, projection, size)
        self.gc2 = ProjectionGraphConvolution(nhid, nclass, projection, size)
        self.dropout = dropout

        print('init ProjectionGCN model done!')

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
