// RUN: hlstool --no_trace --rebuild --tb_file %s dynamic --run_sim

// hlstool will emit a {kernel_name}_tb_output.txt file when sim didn't crash.
// RUN: FileCheck --input-file *tb_output.txt %s

// CHECK: 0

#include "gemver.h"

#ifndef N_KERNEL_CALLS
#define N_KERNEL_CALLS 2
#endif

int main(void) {
  int alpha[N_KERNEL_CALLS];
  int beta[N_KERNEL_CALLS];
  int A[N_KERNEL_CALLS][N][N];
  int u1[N_KERNEL_CALLS][N];
  int v1[N_KERNEL_CALLS][N];
  int u2[N_KERNEL_CALLS][N];
  int v2[N_KERNEL_CALLS][N];
  int w[N_KERNEL_CALLS][N];
  int x[N_KERNEL_CALLS][N];
  int y[N_KERNEL_CALLS][N];
  int z[N_KERNEL_CALLS][N];
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    alpha[i] = rand() % 20;
    beta[i] = rand() % 20;
    for (int yy = 0; yy < N; ++yy) {
      u1[i][yy] = rand() % 20;
      v1[i][yy] = rand() % 20;
      u2[i][yy] = rand() % 20;
      v2[i][yy] = rand() % 20;
      w[i][yy] = rand() % 20;
      x[i][yy] = rand() % 20;
      y[i][yy] = rand() % 20;
      z[i][yy] = rand() % 20;
      for (int x = 0; x < N; ++x) {
        A[i][yy][x] = rand() % 10;
      }
    }
  }
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    gemver(alpha[i], beta[i], A[i], u1[i], v1[i], u2[i], v2[i], w[i], x[i],
           y[i], z[i]);
  }
  return 0;
}