#include "matmul_gpu.h"
#include <numeric> // std::accumulate

// Function to convert a row-major 1D array to a column-major 1D array
void _transpose(const float* row_major, float* column_major, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            column_major[j * rows + i] = row_major[i * cols + j];
        }
    }
}

// GPU / CUDA matrix multiplication
__global__ void cuda_matmul_GPU(const float *A, const float *B, float *C, int m,
                           int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}
