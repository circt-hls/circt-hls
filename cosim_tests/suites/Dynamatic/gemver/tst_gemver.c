#include "gemver.h"

#ifndef AMOUNT_OF_TEST
#define AMOUNT_OF_TEST 1
#endif

int main(void) {
  int alpha[AMOUNT_OF_TEST];
  int beta[AMOUNT_OF_TEST];
  int A[AMOUNT_OF_TEST][N][N];
  int u1[AMOUNT_OF_TEST][N];
  int v1[AMOUNT_OF_TEST][N];
  int u2[AMOUNT_OF_TEST][N];
  int v2[AMOUNT_OF_TEST][N];
  int w[AMOUNT_OF_TEST][N];
  int x[AMOUNT_OF_TEST][N];
  int y[AMOUNT_OF_TEST][N];
  int z[AMOUNT_OF_TEST][N];

  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
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

  // for(int i = 0; i < AMOUNT_OF_TEST; ++i){
  int i = 0;
  gemver(alpha[i], beta[i], A[i], u1[i], v1[i], u2[i], v2[i], w[i], x[i], y[i],
         z[i]);
}
}