import time
import torch
from pytorch_apis import spmm  # Ensure this is correctly installed and imported

# Check if CUDA is available
assert torch.cuda.is_available(), "CUDA is not available."
device = torch.device('cuda')

# Parameters for larger data sizes
num_rows = 1000       # Number of rows in the sparse matrix
num_cols = 1000       # Number of columns in the sparse matrix
num_nonzeros = 5000   # Number of non-zero elements in the sparse matrix
dense_cols = 500      # Number of columns in the dense matrix

# Seed for reproducibility
torch.manual_seed(0)

# Generate random indices for non-zero elements as float32
row_float = torch.randint(0, num_rows, (num_nonzeros,), dtype=torch.int64)
col_float = torch.randint(0, num_cols, (num_nonzeros,), dtype=torch.int64)

# Convert indices to float32
row = row_float.to(torch.float32).to(device)
col = col_float.to(torch.float32).to(device)
indices = torch.stack([row, col], dim=0)

# Generate random values for the non-zero elements
values = torch.randn(num_nonzeros, dtype=torch.float32).to(device)

# Define the size of the sparse matrix
size = torch.Size([num_rows, num_cols])

# Create the sparse matrix in COO format
sparse_matrix = torch.sparse_coo_tensor(indices, values, size).to(device)

# Generate a large dense matrix
dense_matrix = torch.randn(num_cols, dense_cols, dtype=torch.float32).to(device)

# Convert COO to CSR format
# Perform sorting and counting using integer representations
sorted_indices, perm = row_float.sort()
csr_row_int = torch.zeros(num_rows + 1, dtype=torch.int64, device=device)
counts = torch.bincount(sorted_indices, minlength=num_rows)
csr_row_int[1:] = counts
csr_row_int = torch.cumsum(csr_row_int, dim=0)

# Convert csr_row to float32 as per requirement
csr_row = csr_row_int.to(torch.float32).to(device)

# Arrange csr_col and csr_val based on sorted order and convert to float32
csr_col = col_float[perm].to(torch.float32).to(device)
csr_val = values[perm].to(device)

# Perform Sparse Matrix-Dense Matrix Multiplication (SpMM)
start_time = time.time()
result = spmm(csr_row, csr_col, csr_val, dense_matrix, num_rows, dense_cols, device)
end_time = time.time()

# Compute the expected result using PyTorch's built-in operations
expected_result = torch.mm(sparse_matrix.to_dense(), dense_matrix)

# Verify the result
# Since csr_row is float32, but expected_result is accurate, we set a reasonable tolerance
assert torch.allclose(result, expected_result, atol=1e-4), "The spmm result does not match the expected output."
print("SpMM test passed.")
print(f"SpMM completed in {end_time - start_time:.4f} seconds.")
print("Result of spmm (sample):")
print(result[:5, :5].cpu())  # Print a sample of the result
print("\nExpected result (sample):")
print(expected_result[:5, :5].cpu())