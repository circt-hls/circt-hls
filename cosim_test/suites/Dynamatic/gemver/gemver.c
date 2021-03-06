#include "gemver.h"
void gemver(int alpha, int beta, int A[N][N], int u1[N], int v1[N], int u2[N],
            int v2[N], int w[N], int x[N], int y[N], int z[N]) {
  int i, j;
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
  for (i = 0; i < N; i++) {
    int tmp = x[i];
    for (j = 0; j < N; j++)
      tmp = tmp + beta * A[j][i] * y[j];
    x[i] = tmp;
  }
  for (i = 0; i < N; i++)
    x[i] = x[i] + z[i];
  for (i = 0; i < N; i++) {
    int tmp = w[i];
    for (j = 0; j < N; j++)
      tmp = tmp + alpha * A[i][j] * x[j];
    w[i] = tmp;
  }
}