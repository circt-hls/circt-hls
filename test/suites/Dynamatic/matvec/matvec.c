#include "matvec.h"
int matvec(int M[30][30], int V[30], int Out[30]) {
  int i, j;
  int tmp = 0;

  for (i = 0; i < 30; i++) {
    tmp = 0;

    for (j = 0; j < 30; j++) {
      tmp += V[j] * M[i][j];
    }
    Out[i] = tmp;
  }

  return tmp;
}