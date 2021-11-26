
#include "bicg.h"
int bicg(int A[N][N], int s[N], int q[N], int p[N], int r[N]) {
  int i, j;
  int tmp_q = 0;
  for (i = 0; i < NX; i++) {
    tmp_q = q[i];
    for (j = 0; j < NY; j++) {
      int tmp = A[i][j];
      s[j] = s[j] + r[i] * tmp;
      tmp_q = tmp_q + tmp * p[j];
    }
    q[i] = tmp_q;
  }
  return tmp_q;
}
