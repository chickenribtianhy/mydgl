import torch as th
import gp_apis

class spmm_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, input3, input4, dim_0, dim_1, device0):
        res = gp_apis.gp_spmm(input1, input2, input3, input4, dim_0, dim_1, device0)
        # ctx.backward_cache = None #must be implemented
        ctx.save_for_backward(input1, input2, input3, input4)
        ctx.dim_0 = dim_0
        ctx.dim_1 = dim_1
        ctx.device0 = device0
        return res
    

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2, input3, input4 = ctx.saved_tensors
        dim_0 = ctx.dim_0
        dim_1 = ctx.dim_1
        device0 = ctx.device0

        # Create sparse matrix S
        S = th.sparse.FloatTensor(
            th.stack([input1[input2], input2.long()]),
            th,
            th.Size([dim_0, input4.shape[0]])
        ).to(device0)

        grad_dense_matrix = th.sparse.mm(S.t(), grad_output)

        row_indices = input1[input2].long()
        col_indices = input2.long()

        grad_output_rows = grad_output[row_indices, :]  # Shape: (nnz, dim_1)
        dense_matrix_rows = input4[col_indices, :]      # Shape: (nnz, dim_1)

        grad_values = th.sum(grad_output_rows * dense_matrix_rows, dim=1)

        grad_input1 = None
        grad_input2 = None

        # Return gradients
        return grad_input1, grad_input2, grad_values, grad_dense_matrix, None, None, None


def spmm(input1, input2, input3, input4, dim_0, dim_1, device0):        
    # const float *row_ptr, const float *col_idx, const float *values, const float *dense_matrix, float *output,
    return spmm_impl.apply(input1, input2, input3, input4, dim_0, dim_1, device0)
