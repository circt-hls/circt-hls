// RUN: hlstool --no_trace --rebuild --tb_file %s dynamic --run_sim

// hlstool will emit a {kernel_name}_tb_output.txt file when sim didn't crash.
// RUN: FileCheck --input-file *tb_output.txt %s

// CHECK: 0

#include "matrix.h"
#include <stdlib.h>

#ifndef N_KERNEL_CALLS
#define N_KERNEL_CALLS 10
#endif

int main(void) {
  int in_a[N_KERNEL_CALLS][A_ROWS][B_ROWS];
  int in_b[N_KERNEL_CALLS][A_ROWS][B_ROWS];
  int out_c[N_KERNEL_CALLS][A_ROWS][B_ROWS];
  srand(13);
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    for (int y = 0; y < A_ROWS; ++y) {
      for (int x = 0; x < A_ROWS; ++x) {
        in_a[i][y][x] = rand() % 10;
        in_b[i][y][x] = rand() % 10;
      }
    }
  }
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    matrix(in_a[i], in_b[i], out_c[i]);
  }
  return 0;
}
