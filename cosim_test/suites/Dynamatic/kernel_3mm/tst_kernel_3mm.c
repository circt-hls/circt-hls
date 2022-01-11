// RUN: dyn_hlt_lower %s
// RUN: dyn_hlt_build_sim %s
// RUN: dyn_hlt_build_tb %s

#include "kernel_3mm.h"
#ifndef N_KERNEL_CALLS
#define N_KERNEL_CALLS 10
#endif

int main(void) {
  int A[N_KERNEL_CALLS][N][N];
  int B[N_KERNEL_CALLS][N][N];
  int C[N_KERNEL_CALLS][N][N];
  int D[N_KERNEL_CALLS][N][N];
  int E[N_KERNEL_CALLS][N][N];
  int F[N_KERNEL_CALLS][N][N];
  int G[N_KERNEL_CALLS][N][N];
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    for (int y = 0; y < N; ++y) {
      for (int x = 0; x < N; ++x) {
        A[i][y][x] = rand() % 10;
        B[i][y][x] = rand() % 10;
        C[i][y][x] = rand() % 10;
        D[i][y][x] = rand() % 10;
        E[i][y][x] = rand() % 10;
        F[i][y][x] = rand() % 10;
        G[i][y][x] = rand() % 10;
      }
    }
  }
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    kernel_3mm(A[i], B[i], C[i], D[i], E[i], F[i], G[i]);
  }
  return 0;
}