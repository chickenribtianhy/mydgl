#include "kernel.h"
#include <iostream>
#include <stdio.h>
__global__ void spmm_kernel(const float *row_ptr, const float *col_idx, const float *values,
                            const float *dense_matrix, float *output,
                            int num_rows, int num_cols_dense)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("Thread %d processing row %d, num_rows %d\n", threadIdx.x, row, num_rows);
    if (row < num_rows)
    {
        int row_start = (int)(row_ptr[row]);
        int row_end = (int)(row_ptr[row + 1]);
        // printf("Thread %d processing row %d: row_start=%d, row_end=%d\n",
        //    threadIdx.x, row, row_start, row_end);
        for (int idx = row_start; idx < row_end; ++idx)
        {
            int col = col_idx[idx];
            float val = values[idx];

            for (int j = 0; j < num_cols_dense; ++j)
            {
                // Accumulate the multiplication result into the output matrix
                atomicAdd(&output[row * num_cols_dense + j], val * dense_matrix[col * num_cols_dense + j]);
            }
        }
    }
}

void spmm(array1d_t<float> &input1, array1d_t<float> &input2, array1d_t<float> &input3,
          array2d_t<float> &input4, array2d_t<float> &output)
{

    int num_rows = input1.col_count - 1; // Number of rows in the sparse matrix
    int num_cols = input4.col_count;     // Number of columns in the dense matrix

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_rows + threadsPerBlock - 1) / threadsPerBlock;

    // std::cout << num_rows << " " << num_cols << std::endl;

    spmm_kernel<<<blocksPerGrid, threadsPerBlock>>>(input1.data_ptr, input2.data_ptr, input3.data_ptr, input4.data_ptr, output.data_ptr, num_rows, num_cols);
}