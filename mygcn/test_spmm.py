import time

import torch

from pytorch_apis import spmm


assert torch.cuda.is_available()
device = torch.device('cuda')

row = torch.tensor([0, 1, 2], dtype=torch.float32)
col = torch.tensor([0, 1, 2], dtype=torch.float32)
indices = torch.stack([row, col], dim=0)

# Values of the non-zero elements
values = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)

# Size of the sparse matrix (number of rows and columns)
size = torch.Size([3, 3])

# Create the sparse matrix
sparse_matrix = torch.sparse_coo_tensor(indices, values, size).to(device)
# print(sparse_matrix.to_dense())
# Create a dense matrix
dense_matrix = torch.tensor([[1.0, 2.0],
                             [3.0, 4.0],
                             [5.0, 6.0]], dtype=torch.float32).to(device)

# CSR

csr_row = torch.tensor([0, 1, 2, 3], dtype=torch.float32).to(device)
csr_col = col.to(device)
csr_val = values.to(device)
# print(sparse_matrix.size(0), dense_matrix.size(1))
# print(csr_row)
# print(csr_col)
# print(csr_val)
# print(dense_matrix)
result = spmm(csr_row, csr_col, csr_val, dense_matrix, sparse_matrix.size(0), dense_matrix.size(1), device)

expected_result = torch.mm(sparse_matrix.to_dense(), dense_matrix)

# Verify the result
assert torch.allclose(result, expected_result), "The spmm result does not match the expected output."

print("Result of spmm:")
print(result.cpu())
print("\nExpected result:")
print(expected_result.cpu())