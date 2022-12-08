import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.nn as nn
import torch.nn.functional as F


class Graph(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(Graph, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
        # self.weight = Parameter(torch.zeros([in_features, out_features],dtype=torch.float64))
        # if bias:
            # self.bias = Parameter(torch.zeros(out_features,dtype=torch.float64))
        # else:
        
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # print("test")
        # print(self.weight.grad)
        # print(input.float())
        support = torch.mm(input.float(), self.weight.float())
        # print("support")
        # print(support)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return support + self.bias
        else:
            return support

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
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
        # self.weight = Parameter(torch.zeros([in_features, out_features],dtype=torch.float64))
        # if bias:
            # self.bias = Parameter(torch.zeros(out_features,dtype=torch.float64))
        # else:
        
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # print("test")
        # print(self.weight.grad)
        # print(input.float())
        support = torch.mm(input.float(), self.weight.float())
        # print("support")
        # print(support)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output +self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
