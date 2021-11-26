// RUN: dyn_incrementally_lower %s

#include "kernel_3mm.h"
void kernel_3mm(int A[N][N], int B[N][N], int C[N][N], int D[N][N], int E[N][N],
                int F[N][N], int G[N][N]) {
  int i, j, k;
  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++) {
      E[i][j] = 0;
      for (k = 0; k < NK; ++k)
        E[i][j] += A[i][k] * B[k][j];
    }
  for (i = 0; i < NJ; i++)
    for (j = 0; j < NL; j++) {
      F[i][j] = 0;
      for (k = 0; k < NM; ++k)
        F[i][j] += C[i][k] * D[k][j];
    }
  for (i = 0; i < NI; i++)
    for (j = 0; j < NL; j++) {
      G[i][j] = 0;
      for (k = 0; k < NJ; ++k)
        G[i][j] += E[i][k] * F[k][j];
    }
}