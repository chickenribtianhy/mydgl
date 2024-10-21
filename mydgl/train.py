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

import scipy.sparse as sp

from utils import my_load_data, accuracy

from models import GCN

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
# adj: adjacency matrix (e.g., torch sparse tensor or numpy array)
# features: node features (torch.Tensor)
# labels: node labels (torch.Tensor)
# train_mask, test_mask: boolean masks for training and testing nodes (torch.Tensor)

adj, features, labels, train_mask, test_mask = my_load_data()
# adj_csr, features, labels, row_ptr, col_ind, values, train_mask, test_mask
print('Features shape:', features.shape)
print('Labels shape:', labels.shape)
print('Adjacency shape:', adj.shape)
print('Train mask shape:', train_mask.shape)
print('Test mask shape:', test_mask.shape)

# Convert adj to CSR format
def convert_adj_to_csr(adj):
    # If adj is a torch sparse tensor
    if isinstance(adj, torch.sparse.FloatTensor):
        adj = adj.coalesce()
        indices = adj.indices().cpu().numpy()
        values = adj.values().cpu().numpy()
        shape = adj.shape
        adj_csr = sp.csr_matrix((values, (indices[0], indices[1])), shape=shape)
    # If adj is a numpy array or torch dense tensor
    elif isinstance(adj, np.ndarray) or isinstance(adj, torch.Tensor):
        if isinstance(adj, torch.Tensor):
            adj = adj.cpu().numpy()
        adj_csr = sp.csr_matrix(adj)
    # If adj is already a scipy sparse matrix
    elif isinstance(adj, sp.spmatrix):
        adj_csr = adj.tocsr()
    else:
        raise ValueError("Unsupported adjacency matrix format")
    return adj_csr

adj_csr = convert_adj_to_csr(adj)

# Normalize the adjacency matrix (optional)
def normalize_adj(adj_csr):
    """Symmetric normalization of adjacency matrix."""
    adj_csr = adj_csr + sp.eye(adj_csr.shape[0])  # Add self-loops
    degree = np.array(adj_csr.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_normalized = adj_csr.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj_normalized.tocsr()

adj_csr = normalize_adj(adj_csr)

# Extract row_ptr, col_ind, and values from adj_csr
row_ptr = torch.from_numpy(adj_csr.indptr).long()
col_ind = torch.from_numpy(adj_csr.indices).long()
values = torch.from_numpy(adj_csr.data).float()

# Move data to GPU if available
device = torch.device('cuda' if args.cuda else 'cpu')
features = features.to(device)
labels = labels.to(device)
row_ptr = row_ptr.to(device)
col_ind = col_ind.to(device)
values = values.to(device)
train_mask = train_mask.to(device)
test_mask = test_mask.to(device)



# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=int(labels.max()) + 1,
            dropout=args.dropout)
model = model.to(device)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

# Training function
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, row_ptr, col_ind, values, adj_csr.shape, device)
    loss_train = F.nll_loss(output[train_mask], labels[train_mask])
    acc_train = accuracy(output[train_mask], labels[train_mask])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate on training data
        model.eval()
        output = model(features, row_ptr, col_ind, values, adj_csr.shape, device)

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'time: {:.4f}s'.format(time.time() - t))

# Testing function
def test():
    model.eval()
    output = model(features, row_ptr, col_ind, values, adj_csr.shape, device)
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