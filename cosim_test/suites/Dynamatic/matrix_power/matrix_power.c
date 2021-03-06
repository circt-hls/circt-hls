#include "matrix_power.h"
void matrix_power(int x[20][20], int row[20], int col[20], int a[20]) {
  for (int k = 1; k < 20; k++) {
    for (int p = 0; p < 20; p++) {
      x[k][row[p]] += a[p] * x[k - 1][col[p]];
    }
  }
}
