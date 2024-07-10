#include <arm_neon.h>
#include <numeric> // std::accumulate

// Function to convert a row-major 1D array to a column-major 1D array
void _transpose(const float *row_major, float *column_major, int rows, int cols)
{
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      column_major[j * rows + i] = row_major[i * cols + j];
    }
  }
}

// NEON matrix multiplication
// NEON matrix multiplication
void matmul_NEON(const float *A, const float *B_row, float *C, int m, int n, int k)
{
  float *B = new float[n * k];
  _transpose(B_row, B, n, k);

  // Determine NEON register size (assuming float32x4_t)
  const int register_size = 4;

  // Iterate over each row of C (m rows)
  for (int i = 0; i < m; ++i)
  {
    // Iterate over each column of C (k columns)
    for (int j = 0; j < k; ++j)
    {
      // Initialize the result to 0
      float32x4_t sum = vdupq_n_f32(0.0f);

      // Iterate over each element of the row of A and the column of B
      int l = 0;
      for (; l <= n - register_size; l += register_size)
      {
        // Load the elements of A and B into NEON registers
        // vld4q_u32 loads 8
        //
        float32x4_t a = vld1q_f32(A + i * n + l);
        float32x4_t b = vld1q_f32(B + j * n + l);

        // Multiply the elements and add to the sum
        sum = vmlaq_f32(sum, a, b);
      }

      // Handle remaining elements
      float remaining_sum = 0.0f;
      for (; l < n; ++l)
      {
        remaining_sum += A[i * n + l] * B[j * n + l];
      }

      // Store the result in C
      float temp[register_size];
      vst1q_f32(temp, sum);

      float total_sum = std::accumulate(temp, temp + register_size, remaining_sum);
      C[i * k + j] = total_sum;
    }
  }

  delete[] B;
}