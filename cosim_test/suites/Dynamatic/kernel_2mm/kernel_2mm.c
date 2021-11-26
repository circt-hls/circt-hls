// RUN: dyn_incrementally_lower %s

#include "kernel_2mm.h"
void kernel_2mm(int alpha, int beta, int tmp[N][N], int A[N][N], int B[N][N],
                int C[N][N], int D[N][N]) {
  int i, j, k;

  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++) {
      tmp[i][j] = 0;
      for (k = 0; k < NK; ++k)
        tmp[i][j] += alpha * A[i][k] * B[k][j];
    }
  for (i = 0; i < NI; i++)
    for (j = 0; j < NL; j++) {
      D[i][j] *= beta;
      for (k = 0; k < NJ; ++k)
        D[i][j] += tmp[i][k] * C[k][j];
    }
}