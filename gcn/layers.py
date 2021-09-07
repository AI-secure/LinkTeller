import math

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class ProjectionGraphConvolution(Module):
    def __init__(self, in_features, out_features, projection, size, bias=True, trainable=False, args=None):
        super(ProjectionGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # self.mask = (projection != 0).type(torch.FloatTensor)
        if torch.cuda.is_available():
            # self.mask = self.mask.cuda()
            self.projection = projection.cuda()
        # self.projection = Parameter(projection)

        self.n_nodes = size
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.coeff = Parameter(torch.FloatTensor([1]), requires_grad=trainable)
        self.reset_parameters()

    def reset_parameters(self):
        # torch.manual_seed(100)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        # self.coeff.data.uniform_(0.5, 5)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)

        output = torch.spmm(self.projection, support)
        output = torch.spmm(adj, output) + self.coeff * support

        # output = (1+self.coeff) * support
        # output = torch.spmm(adj, output) + support
        # adjacency = torch.spmm(adj, self.projection*self.mask) + self.identity
        # adjacency = torch.spmm(adj, self.projection) + self.identity

        # adjacency = torch.spmm(adj, self.projection)
        # output = torch.spmm(adjacency, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class BasicMLP(Module):
    def __init__(self, in_features, out_features, size, bias=True, args=None, trainable=False):
        super(BasicMLP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.n_nodes = size
        self.coeff = Parameter(torch.FloatTensor([1]), requires_grad=trainable)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, d_vec):
        support = torch.mm(input, self.weight)

        output = torch.matmul(torch.ones(self.n_nodes,1).cuda(),
                            torch.matmul(torch.ones(1,self.n_nodes).cuda(), support))# + support

        # output = torch.mm(torch.ones(self.n_nodes, self.n_nodes).cuda()/2708, self.coeff * support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class DegreeMLP(Module):
    def __init__(self, in_features, out_features, size, bias=True, args=None, trainable=False):
        super(DegreeMLP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.projection = torch.ones(1, size).cuda() / size

        self.n_nodes = size
        # self.d_vec_add = Parameter(torch.FloatTensor(n_nodes), requires_grad=False)
        self.d_vec_add = Parameter(torch.zeros(self.n_nodes), requires_grad=True)
        self.coeff = Parameter(torch.FloatTensor([args.coeff]), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.d_vec_add.data.uniform_(-1, 1)

    def forward(self, input, d_vec):
        support = torch.mm(input, self.weight)
        output = torch.spmm(self.projection, support)

        output = torch.mm(
            (torch.ones(self.n_nodes).cuda()+self.d_vec_add).unsqueeze(-1), output)# + self.coeff * support

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'