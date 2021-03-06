// This circuit deadlocks if we use buffer size <=2
// RUN: hlstool --no_trace --rebuild --tb_file %s dynamic-polygeist --run_sim

// hlstool will emit a {kernel_name}_tb_output.txt file when sim didn't crash.
// RUN: FileCheck --input-file *tb_output.txt %s

// CHECK: 0
// --buffer_size=3

#include "matrix_power.h"
#include <stdlib.h>

#ifndef N_KERNEL_CALLS
#define N_KERNEL_CALLS 10
#endif

int main(void) {
  int mat[N_KERNEL_CALLS][20][20];
  int row[N_KERNEL_CALLS][20];
  int col[N_KERNEL_CALLS][20];
  int a[N_KERNEL_CALLS][20];
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    for (int y = 0; y < 20; ++y) {
      col[i][y] = rand() % 20;
      row[i][y] = rand() % 20;
      a[i][y] = rand();
      for (int x = 0; x < 20; ++x) {
        mat[i][y][x] = 0;
      }
    }
  }
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    matrix_power(mat[i], row[i], col[i], a[i]);
  }
  return 0;
}
