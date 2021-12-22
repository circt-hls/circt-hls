// RUN: hlstool --no_trace --rebuild --tb_file %s dynamic --run_sim

// hlstool will emit a {kernel_name}_tb_output.txt file when sim didn't crash.
// RUN: FileCheck --input-file *tb_output.txt %s

// CHECK: 0

#include "kernel_2mm.h"
#ifndef N_KERNEL_CALLS
#define N_KERNEL_CALLS 1
#endif

int main(void) {
  int alpha[N_KERNEL_CALLS];
  int beta[N_KERNEL_CALLS];
  int tmp[N_KERNEL_CALLS][N][N];
  int A[N_KERNEL_CALLS][N][N];
  int B[N_KERNEL_CALLS][N][N];
  int C[N_KERNEL_CALLS][N][N];
  int D[N_KERNEL_CALLS][N][N];
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    alpha[i] = rand();
    beta[i] = rand();
    for (int y = 0; y < N; ++y) {
      for (int x = 0; x < N; ++x) {
        A[i][y][x] = rand() % 100;
        B[i][y][x] = rand() % 100;
        C[i][y][x] = rand() % 100;
        D[i][y][x] = rand() % 100;
      }
    }
  }
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    kernel_2mm(alpha[i], beta[i], tmp[i], A[i], B[i], C[i], D[i]);
  }
  return 0;
}