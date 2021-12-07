// RUN: hlstool --tb_file %s dynamic

#include "kernel_2mm.h"
#ifndef AMOUNT_OF_TEST
#define AMOUNT_OF_TEST 1
#endif
int main(void) {
  int alpha[AMOUNT_OF_TEST];
  int beta[AMOUNT_OF_TEST];
  int tmp[AMOUNT_OF_TEST][N][N];
  int A[AMOUNT_OF_TEST][N][N];
  int B[AMOUNT_OF_TEST][N][N];
  int C[AMOUNT_OF_TEST][N][N];
  int D[AMOUNT_OF_TEST][N][N];
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
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
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    kernel_2mm(alpha[0], beta[0], tmp[0], A[0], B[0], C[0], D[0]);
  }
}