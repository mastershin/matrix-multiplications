/*
Multiplies Matrices A, B, C with sizes m x k, k x n, m x n)

A x B = C
*/

#include <cassert>
#include <chrono>
#include <iostream>
#include <tuple>
#include <cmath>

#if !defined(arm64) && !defined(__aarch64__)
#error "This code is for ARM64 only"
#endif

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#elif defined(__linux__) || defined(__APPLE__)
#include <stdlib.h>
#include <sys/mman.h>
#endif

#include "main.h"
#include "neon.h"
#include "matmul_cpu_neon.h"
#include "test_matmul.h"

int num_loops = 1;
#define TOLERANCE 1e-5 // Tolerance for floating-point comparison

using namespace std;
using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

time_point now()
{
  return std::chrono::high_resolution_clock::now();
}

auto time_diff(time_point a, time_point b)
{
  // Return Type: std::chrono::duration<double>
  return std::chrono::duration_cast<std::chrono::duration<double>>(b - a);
}

void initialize_data(float *&A, float *&B, float *&C, int &m, int &n, int &k)
{
  // Matrix Multiplication: A * B = C

  // Calculate the size of matrices A, B, and C
  int sizeA = m * n;
  int sizeB = n * k;
  int sizeC = m * k;

  // Allocate memory for the matrices dynamically
  A = new float[sizeA];
  B = new float[sizeB];
  C = new float[sizeC]{0.0f};

  // Initialize matrices A and B with sequential float numbers
  for (int i = 0; i < sizeA; ++i)
  {
    if (i % 2 == 0)
      A[i] = (float)i; // Assign sequential float values
    else
      A[i] = (float)-i;
  }

  for (int i = 0; i < sizeB; ++i)
  {
    if (i % 3 == 0)
      B[i] = 1.0f; // Assign sequential float values
    else if (i % 3 == 1)
      B[i] = 2.0f;
    else
      B[i] = 3.0f;
  }
}

void get_small_matrix_size(int &m, int &n, int &k)
{
  // (200x150, 150x100 --> 200x100)
  // m=200, n=150, k=100 --> sum --> -200 (~1.3 sec on CPU)
  m = 200;
  n = 150;
  k = 100;
}

void get_medium_matrix_size(int &m, int &n, int &k)
{
  // 500x300, 300x200, 500x200 --> sum --> -400 (~13 sec on CPU)
  m = 500;
  n = 300;
  k = 200;
}

void get_large_matrix_size(int &m, int &n, int &k)
{
  // More realistic small-scale LLM size
  // 4096x1024, 1024x1024, 4096x1024 --> sum --> ?
  // int m = 4096, n = 1024, k = 1024;

  m = 4096;
  n = 1024;
  k = 1024;
}

bool verify_result(const float *C1, const float *C2, int m, int n, int k)
{
  for (int i = 0; i < m * k; ++i)
  {
    if (std::fabs(C1[i] - C2[i]) > TOLERANCE)
    {
      std::cerr << C1[i] << " vs " << C2[i] << " at i=" << i << std::endl;
      return false;
    }
  }
  return true;
}

// Classic CPU matrix multiplication using for loop (slow)
void matmul_cpu(const float *A, const float *B, float *C, int m, int n, int k)
{
  for (int i = 0; i < m; ++i)
  {
    for (int j = 0; j < k; ++j)
    {
      C[i * k + j] = 0.0f;
      for (int p = 0; p < n; ++p)
      {
        C[i * k + j] += A[i * n + p] * B[p * k + j];
      }
    }
  }
}

void _verify_matmul_neon(int m, int n, int k)
{
  // Allocate memory for matrices A, B, and C
  float *A, *B, *C, *C_cpu;

  // Initialize vectors a and b
  initialize_data(A, B, C, m, n, k);
  C_cpu = new float[m * n]{0.0f};

  // Perform CPU matrix multiplication for verification
  matmul_cpu(A, B, C_cpu, m, n, k);

  // Verify AVX (128 bit) result against CPU result
  // if (__builtin_cpu_supports("avx2")) {
  if (detect_neon())
  {
    float *C_neon = new float[m * n]{0.0f};
    matmul_NEON(A, B, C_neon, m, n, k);
    if (!verify_result(C_cpu, C_neon, m, n, k))
    {
      std::cerr << "Verification failed for NEON" << std::endl;
      delete[] A;
      delete[] B;
      delete[] C;
      delete[] C_cpu;
      delete[] C_neon;
      exit(1);
    }
    delete[] C_neon;
    std::cout << "Verification Successful for NEON" << std::endl;
  }

  std::cout << "Verification passed" << std::endl;
}

int do_matmul_neon(std::string neon_type, int m, int n, int k)
{

  // Allocate memory for matrices A, B, and C
  float *A, *B, *C;

  // Initialize vectors a and b
  initialize_data(A, B, C, m, n, k);

  auto start_cpu = now();

  std::cout << neon_type << ":";
  for (int i = 0; i < num_loops; i++)
  {
    std::cout << "." << std::flush;
    if (neon_type == "neon")
    {
      matmul_NEON(A, B, C, m, n, k);
    }
  }
  auto end_cpu = now();
  std::chrono::duration<double> duration = end_cpu - start_cpu;

  std::cout << std::endl;
  std::cout << neon_type << " CPU time: " << duration.count() << " seconds"
            << std::endl;

  // Clean up
  delete[] A;
  delete[] B;
  delete[] C;

  return 0;
}

std::tuple<int, int, int> process_commands(const ArgsMap &args_map)
{
  int m = 0, n = 0, k = 0;

  string matrix_size = args_map.at("size");

  if (matrix_size == "s")
  {
    get_small_matrix_size(m, n, k);
  }
  else if (matrix_size == "m")
  {
    get_medium_matrix_size(m, n, k);
  }
  else if (matrix_size == "l")
  {
    get_large_matrix_size(m, n, k);
  }
  else
  {
    die("Invalid size argument. Use 's', 'm', 'l'.");
  }

  num_loops = stoi(args_map.at("loop"));
  cout << "Number of loops: " << num_loops << endl;

  return std::make_tuple(m, n, k);
}

int main(int argc, char *argv[])
{
  print_neon_features();
  if (!detect_neon())
  {
    die("NEON is not supported on this device");
  }

  auto args_map = parse_command_args(argc, argv);
  auto [m, n, k] = process_commands(args_map);

  std::cout << "Matrix Multiplication: A(" << m << "x" << k << ") * B(" << k
            << "x" << n << ") = C(" << m << "x" << n << ")" << std::endl;

  std::cout << "Testing matmul_cpu" << std::endl;
  test_matmul(matmul_cpu);

  std::cout << "Testing matmul_NEON" << std::endl;
  test_matmul(matmul_NEON);

  // one first testing with NEON vs non-NEON results
  _verify_matmul_neon(m, n, k);

  do_matmul_neon("neon", m, n, k);

  return 0;
}