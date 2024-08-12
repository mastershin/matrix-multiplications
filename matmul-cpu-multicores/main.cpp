/*
Multiplies Matrices A, B, C with sizes m x k, k x n, m x n)

A x B = C

*/

#include "main.h"
#include "matmul_cpu_multicores.h"
#include "test_matmul.h"
#include <thread>

#define TOLERANCE 1e-5 // Tolerance for floating-point comparison

using namespace std;
using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

int num_loops = 1;

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
  int sizeA = m * k;
  int sizeB = k * n;
  int sizeC = m * n;

  // Allocate memory for the matrices dynamically
  A = new float[sizeA];
  B = new float[sizeB];
  C = new float[sizeC]{0.0f};

  // Initialize matrices A and B with sequential float numbers
  for (int i = 0; i < sizeA; ++i)
  {
    if (i % 2 == 0)
      A[i] = i; // Assign sequential float values
    else
      A[i] = -i;
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
      return false;
    }
  }
  return true;
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
  auto args_map = parse_command_args(argc, argv);
  if (args_map.size() < 2)
  {
    die("Usage: ./main --size [s|m|l] --loop [num_loops]");
  }
  auto [m, n, k] = process_commands(args_map);

  std::cout << "Matrix Multiplication: A(" << m << "x" << k << ") * B(" << k
            << "x" << n << ") = C(" << m << "x" << n << ")" << std::endl;

  int num_threads = std::thread::hardware_concurrency();
  std::cout << "Number of threads: " << num_threads << std::endl;

  test_matmul(matmul_cpu_multicores);

  // Allocate memory for matrices A, B, and C
  float *A, *B, *C;

  // Initialize vectors a and b
  initialize_data(A, B, C, m, n, k);

  // CPU vector addition
  cout << "Starting the loop..." << endl;
  auto start_cpu = now();

  for (int i = 0; i < num_loops; i++)
  {
    std::cout << "." << std::flush;
    matmul_cpu_multicores(A, B, C, m, n, k);
  }
  auto end_cpu = now();
  std::chrono::duration<double> duration = end_cpu - start_cpu;

  cout << std::endl;
  cout << "CPU time: " << duration.count() << " seconds" << endl;

  float sum = 0.0;
  for (int i = 0; i < m * k; ++i)
  {
    sum += C[i];
  }
  cout << "Sum: " << sum << endl;

  // Clean up
  delete[] A;
  delete[] B;
  delete[] C;

  return 0;
}