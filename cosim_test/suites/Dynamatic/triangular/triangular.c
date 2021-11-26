#include "triangular.h"
void triangular(int x[10], int A[10][10], int n) {
  for (int i = n - 1; i >= 0; i--) {
    // x[i] = A[i][n]/A[i][i];
    for (int k = i - 1; k >= 0; k--) {
      A[k][n] -= A[k][i] * x[i];
    }
  }
}
