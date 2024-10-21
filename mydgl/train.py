from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import my_load_data, accuracy


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
# The data returned by my_load_data():
# adj: adjacency matrix (probably in sparse format)
# features: node features
# labels: node labels
# row_ptr, col_ind: CSR representation of adj (may not be needed here)
# train_mask, test_mask: boolean masks for training and testing

adj, features, labels, row_ptr, col_ind, train_mask, test_mask = my_load_data()
print('Features shape:', features.shape)
print('Labels shape:', labels.shape)
print('Adjacency shape:', adj.shape)
print('Train mask shape:', train_mask.shape)
print('Test mask shape:', test_mask.shape)

# Convert adjacency matrix to sparse tensor if it's not already
if not isinstance(adj, torch.sparse.FloatTensor):
    # Assuming adj is in scipy sparse format
    import scipy.sparse as sp
    if isinstance(adj, sp.spmatrix):
        adj = adj.tocoo()
        indices = torch.from_numpy(
            np.vstack((adj.row, adj.col)).astype(np.int64))
        values = torch.from_numpy(adj.data.astype(np.float32))
        shape = torch.Size(adj.shape)
        adj = torch.sparse.FloatTensor(indices, values, shape)
    else:
        # If adj is a dense numpy array
        adj = torch.FloatTensor(adj)
        # Convert to sparse tensor
        adj = adj.to_sparse()

# Normalize adjacency matrix (if needed)
def normalize_adj(adj):
    """Row-normalize sparse matrix"""
    adj = adj.coalesce()
    indices = adj.indices()
    values = adj.values()
    row_sum = torch.sparse.sum(adj, dim=1).to_dense()
    inv_row_sum = 1.0 / row_sum
    inv_row_sum[inv_row_sum == float('inf')] = 0.0
    D_inv = torch.diag(inv_row_sum)
    adj_normalized = torch.sparse.mm(adj, D_inv)
    return adj_normalized

adj = normalize_adj(adj)

# Move data to GPU if available
if args.cuda:
    features = features.cuda()
    labels = labels.cuda()
    adj = adj.cuda()
    train_mask = train_mask.cuda()
    test_mask = test_mask.cuda()

# Define the GCN model using your custom GraphConvolution layer
from models import GCN

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=int(labels.max()) + 1,
            dropout=args.dropout)

if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

# Training function
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[train_mask], labels[train_mask])
    acc_train = accuracy(output[train_mask], labels[train_mask])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate on training data
        model.eval()
        output = model(features, adj)

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'time: {:.4f}s'.format(time.time() - t))

# Testing function
def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[test_mask], labels[test_mask])
    acc_test = accuracy(output[test_mask], labels[test_mask])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()