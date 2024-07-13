#pragma once

#include <cuda_runtime.h>
#include <iostream>

#define gpuErrorCheck(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

//__host__ void stub_matmul_GPU(const vector<float> &A, const vector<float> &B, vector<float> &C, int m, int n, int k);
__global__ void cuda_matmul_GPU(const float* A, const float* B_row, float* C,
                                int m, int n, int k);
