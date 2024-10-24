import math
import numpy as np

import torch
import torch.nn as nn

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


from pytorch_apis import spmm

class GraphConvolution(nn.Module):
    """
    Simple GCN layer using custom spmm function.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, row_ptr, col_ind, values, adj_shape, device):
        support = torch.mm(input, self.weight)
        # Prepare inputs for spmm function
        dim_0 = adj_shape[0]
        dim_1 = support.shape[1]
        output = spmm(row_ptr, col_ind, values, support, dim_0, dim_1, device)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return '{} ({} -> {})'.format(self.__class__.__name__,
                                      self.in_features,
                                      self.out_features)
