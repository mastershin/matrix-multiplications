/*
Multiplies Matrices A, B, C with sizes m x k, k x n, m x n)

A x B = C
*/

#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>

#if defined(arm64) || defined(__aarch64__) || defined(__APPLE__)
#error "This code is for x86/x64 CUDA only"
#endif

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#elif defined(__linux__)
#include <stdlib.h>
#include <sys/mman.h>
#endif

#include "main.h"
#include "matmul_gpu.h"
#include "test_matmul.h"

int num_loops = 1;
#define TOLERANCE 1e-5  // Tolerance for floating-point comparison

using namespace std;
using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

time_point now() {
  return std::chrono::high_resolution_clock::now();
}

auto time_diff(time_point a, time_point b) {
  // Return Type: std::chrono::duration<double>
  return std::chrono::duration_cast<std::chrono::duration<double>>(b - a);
}

void initialize_data(vector<float>& A, vector<float>& B, vector<float>& C,
                     int m, int n, int k) {
  std::cout << "Data Init Begin" << std::endl;
  // Matrix Multiplication: A * B = C

  // Calculate the size of matrices A, B, and C
  int sizeA = m * n;
  int sizeB = n * k;
  int sizeC = m * k;

  // Initialize matrices A and B with sequential float numbers
  for (int i = 0; i < sizeA; ++i) {
    if (i % 2 == 0)
      A[i] = (float)i;  // Assign sequential float values
    else
      A[i] = (float)-i;
  }

  for (int i = 0; i < sizeB; ++i) {
    if (i % 3 == 0)
      B[i] = 1.0f;  // Assign sequential float values
    else if (i % 3 == 1)
      B[i] = 2.0f;
    else
      B[i] = 3.0f;
  }
  std::cout << "Data init done." << std::endl;
}

void get_small_matrix_size(int& m, int& n, int& k) {
  // (200x150, 150x100 --> 200x100)
  // m=200, n=150, k=100 --> sum --> -200 (~1.3 sec on CPU)
  m = 200;
  n = 150;
  k = 100;
}

void get_medium_matrix_size(int& m, int& n, int& k) {
  // 500x300, 300x200, 500x200 --> sum --> -400 (~13 sec on CPU)
  m = 500;
  n = 300;
  k = 200;
}

void get_large_matrix_size(int& m, int& n, int& k) {
  // More realistic small-scale LLM size
  // 4096x1024, 1024x1024, 4096x1024 --> sum --> ?
  // int m = 4096, n = 1024, k = 1024;

  m = 4096;
  n = 1024;
  k = 1024;
}

void get_xl_matrix_size(int& m, int& n, int& k) {
  // 2 billion parameter LLMs
  m = 4096;
  n = 8192;
  k = 16384;
}

void get_xxl_matrix_size(int& m, int& n, int& k) {
  // Uses about 1 GB of GPU memory, 256 million x 256 million matrix
  m = 16384;
  n = 16384;
  k = 16384;
}

bool verify_result(const float* C1, const float* C2, int m, int n, int k) {
  for (int i = 0; i < m * k; ++i) {
    if (std::fabs(C1[i] - C2[i]) > TOLERANCE) {
      std::cerr << C1[i] << " vs " << C2[i] << " at i=" << i << std::endl;
      return false;
    }
  }
  return true;
}

// Classic CPU matrix multiplication using for loop (slow)
// void matmul_cpu(const float *A, const float *B, float *C, int m, int n, int
// k) {
void matmul_cpu(const vector<float> A, const vector<float> B, vector<float> C,
                int m, int n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < k; ++j) {
      C.at(i * k + j) = 0.0f;
      for (int p = 0; p < n; ++p) {
        C.at(i * k + j) += A[i * n + p] * B[p * k + j];
      }
    }
  }
}

void stub_matmul_GPU_float(const float* h_A, const float* h_B, float* h_C,
                           int m, int n, int k) {
  float* d_A;
  float* d_B;
  float* d_C;
  int sizeA = m * n;
  int sizeB = n * k;
  int sizeC = m * k;
  gpuErrorCheck(cudaMalloc(&d_A, sizeA * sizeof(float)));
  gpuErrorCheck(cudaMalloc(&d_B, sizeB * sizeof(float)));
  gpuErrorCheck(cudaMalloc(&d_C, sizeC * sizeof(float)));

  // Copy data to device
  gpuErrorCheck(
      cudaMemcpy(d_A, h_A, sizeA * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrorCheck(
      cudaMemcpy(d_B, h_B, sizeB * sizeof(float), cudaMemcpyHostToDevice));

  cuda_matmul_GPU(d_A, d_B, d_C, m, n, k);
  cudaDeviceSynchronize();

  // copy from GPU to host
  gpuErrorCheck(
      cudaMemcpy(h_C, d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
void stub_matmul_GPU(vector<float>& A, vector<float>& B, vector<float>& C,
                     int m, int n, int k) {
  float* d_C;
  gpuErrorCheck(cudaMalloc(&d_C, C.size() * sizeof(float)));

  cuda_matmul_GPU(&A.at(0), &B.at(0), d_C, m, n, k);
  cudaDeviceSynchronize();

  // copy from GPU to host
  gpuErrorCheck(cudaMemcpy(&C.at(0), d_C, C.size() * sizeof(float),
                           cudaMemcpyDeviceToHost));

  assert(C.size() == m * k);
  // C.assign(d_C, d_C + C.size());
  cudaFree(d_C);
}

void _verify_matmul_GPU(vector<float>& A, vector<float>& B, vector<float>& C,
                        int m, int n, int k) {
  // Allocate memory for matrices A, B, and C

  // Initialize vectors a and b
  initialize_data(A, B, C, m, n, k);
  vector<float> C_cpu(C.size(), 0);

  // Perform CPU matrix multiplication for verification
  matmul_cpu(A, B, C_cpu, m, n, k);

  stub_matmul_GPU(A, B, C, m, n, k);

  if (!verify_result(&C_cpu.at(0), &C.at(0), m, n, k)) {
    std::cerr << "Verification failed for GPU/CUDA" << std::endl;
    exit(1);
  }

  std::cout << "Verification Successful for GPU/CUDA" << std::endl;

  std::cout << "Verification passed" << std::endl;
}

void do_matmul_GPU(vector<float>& A, vector<float>& B, vector<float>& C, int m,
                   int n, int k) {
  // Initialize vectors a and b
  initialize_data(A, B, C, m, n, k);

  auto start_cpu = now();

  for (int i = 0; i < num_loops; i++) {
    std::cout << "." << std::flush;
    stub_matmul_GPU(A, B, C, m, n, k);
  }
  auto end_cpu = now();
  std::chrono::duration<double> duration = end_cpu - start_cpu;

  std::cout << std::endl;
  std::cout << "GPU/CUDA time: " << duration.count() << " seconds" << std::endl;
}

std::tuple<int, int, int> process_commands(const ArgsMap& args_map) {
  int m = 0, n = 0, k = 0;

  string matrix_size = args_map.at("size");

  if (matrix_size == "s") {
    get_small_matrix_size(m, n, k);
  } else if (matrix_size == "m") {
    get_medium_matrix_size(m, n, k);
  } else if (matrix_size == "l") {
    get_large_matrix_size(m, n, k);
  } else if (matrix_size == "xl") {
    get_xl_matrix_size(m, n, k);
  } else if (matrix_size == "xxl") {
    get_xxl_matrix_size(m, n, k);
  } else {
    die("Invalid size argument. Use 's', 'm', 'l'.");
  }

  num_loops = stoi(args_map.at("loop"));
  cout << "Number of loops: " << num_loops << endl;

  return std::make_tuple(m, n, k);
}

void check_cuda() {
  // Choose which GPU to run on, change this on a multi-GPU system.
  int device_id = 0;
  cudaError_t cudaStatus;
  cudaStatus = cudaSetDevice(device_id);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr,
            "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    exit(1);
  }
  std::cout << "CUDA OK. Using device: " << device_id << std::endl;
}
int main(int argc, char* argv[]) {
  auto args_map = parse_command_args(argc, argv);
  auto [m, n, k] = process_commands(args_map);

  vector<float> A(m * n);
  vector<float> B(n * k);
  vector<float> C(m * k);

  std::cout << "Matrix Multiplication: A(" << m << "x" << n << ") * B(" << n
            << "x" << k << ") = C(" << m << "x" << k << ")" << std::endl;
  std::cout << "sizeA=" << A.size() << " sizeB=" << B.size()
            << " sizeC=" << C.size() << std::endl;

  // std::cout << "Testing matmul_GPU / CUDA" << std::endl;
  // test_matmul(stub_matmul_GPU_float);

  // one last testing with GPU vs CPU results for large matrix
  // really slow for Large Matrix.
  // std::cout << "Testing GPU vs CPU result" << std::endl;
  //_verify_matmul_GPU(A, B, C, m, n, k);

  std::cout << "Actual GPU testing..." << std::endl;
  do_matmul_GPU(A, B, C, m, n, k);

  return 0;
}