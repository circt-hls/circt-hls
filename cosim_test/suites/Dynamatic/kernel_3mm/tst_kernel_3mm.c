// RUN: dyn_hlt_lower %s
// RUN: dyn_hlt_build_sim %s
// RUN: dyn_hlt_build_tb %s

#include "kernel_3mm.h"
#ifndef AMOUNT_OF_TEST
#define AMOUNT_OF_TEST 1
#endif
int main(void) {
  int A[AMOUNT_OF_TEST][N][N];
  int B[AMOUNT_OF_TEST][N][N];
  int C[AMOUNT_OF_TEST][N][N];
  int D[AMOUNT_OF_TEST][N][N];
  int E[AMOUNT_OF_TEST][N][N];
  int F[AMOUNT_OF_TEST][N][N];
  int G[AMOUNT_OF_TEST][N][N];
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
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
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    kernel_3mm(A[0], B[0], C[0], D[0], E[0], F[0], G[0]);
  }
}