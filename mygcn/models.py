import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(GCN, self).__init__()

#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, nclass)
#         self.dropout = dropout

#     def forward(self, x, adj, row_ptr, col_ind, values, adj_shape, device):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.relu(self.gc1(x, adj, row_ptr, col_ind, values, adj_shape, device))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, adj, row_ptr, col_ind, values, adj_shape, device)
#         return F.log_softmax(x, dim=1)


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
