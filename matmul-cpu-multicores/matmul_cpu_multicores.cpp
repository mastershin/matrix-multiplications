#include <thread>
#include <vector>

void matmul_worker(const float *A, const float *B, float *C, int m, int n,
                   int k, int start_row, int end_row) {
  for (int i = start_row; i < end_row; ++i) {
    for (int j = 0; j < k; ++j) {
      C[i * k + j] = 0.0f;
      for (int p = 0; p < n; ++p) {
        C[i * k + j] += A[i * n + p] * B[p * k + j];
      }
    }
  }
}

void matmul_cpu_multicores(const float *A, const float *B, float *C, int m, int n, int k) {
  int num_threads = std::thread::hardware_concurrency();

  std::vector<std::thread> threads;

  int rows_per_thread = m / num_threads;
  int extra_rows = m % num_threads;

  int start_row = 0;
  for (int t = 0; t < num_threads; ++t) {
    int end_row = start_row + rows_per_thread + (t < extra_rows ? 1 : 0);
    threads.emplace_back(matmul_worker, A, B, C, m, n, k, start_row, end_row);
    start_row = end_row;
  }

  for (auto &th : threads) {
    th.join();
  }
}
