// RUN: hlstool --no_trace --rebuild --tb_file %s dynamic-polygeist --run_sim

// hlstool will emit a {kernel_name}_tb_output.txt file when sim didn't crash.
// RUN: FileCheck --input-file *tb_output.txt %s

// CHECK: 0

#include "bicg.h"
#include <stdlib.h>

#ifndef N_KERNEL_CALLS
#define N_KERNEL_CALLS 10
#endif

int main(void) {
  int A[N_KERNEL_CALLS][N][N];
  int s[N_KERNEL_CALLS][N];
  int q[N_KERNEL_CALLS][N];
  int p[N_KERNEL_CALLS][N];
  int r[N_KERNEL_CALLS][N];
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    for (int y = 0; y < N; ++y) {
      s[i][y] = rand() % 100;
      q[i][y] = rand() % 100;
      p[i][y] = rand() % 100;
      r[i][y] = rand() % 100;
      for (int x = 0; x < N; ++x) {
        A[i][y][x] = rand() % 100;
      }
    }
  }
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    bicg(A[i], s[i], q[i], p[i], r[i]);
  }
  return 0;
}
