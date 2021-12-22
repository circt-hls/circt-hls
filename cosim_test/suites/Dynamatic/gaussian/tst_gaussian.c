// RUN: hlstool --no_trace --rebuild --tb_file %s dynamic --run_sim

// hlstool will emit a {kernel_name}_tb_output.txt file when sim didn't crash.
// RUN: FileCheck --input-file *tb_output.txt %s

// CHECK: 0

#include "gaussian.h"
#include <stdlib.h>

#ifndef N_KERNEL_CALLS
#define N_KERNEL_CALLS 1
#endif

int main(void) {
  int c[N_KERNEL_CALLS][20];
  int A[N_KERNEL_CALLS][20][20];
  srand(13);
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    for (int y = 0; y < 20; ++y) {
      c[i][y] = 1; // rand()%20;
      for (int x = 0; x < 20; ++x) {
        A[i][y][x] = 1; // rand()%20;
      }
    }
  }
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    gaussian(c[i], A[i]);
  }
  return 0;
}
