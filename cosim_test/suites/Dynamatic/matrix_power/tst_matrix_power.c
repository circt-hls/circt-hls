// RUN: dyn_hlt_lower %s
// RUN: dyn_hlt_build_sim %s
// RUN: dyn_hlt_build_tb %s

#include "matrix_power.h"
#include <stdlib.h>

#ifndef AMOUNT_OF_TEST
#define AMOUNT_OF_TEST 1
#endif
int main(void) {
  int mat[AMOUNT_OF_TEST][20][20];
  int row[AMOUNT_OF_TEST][20];
  int col[AMOUNT_OF_TEST][20];
  int a[AMOUNT_OF_TEST][20];
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    for (int y = 0; y < 20; ++y) {
      col[i][y] = rand() % 20;
      row[i][y] = rand() % 20;
      a[i][y] = rand();
      for (int x = 0; x < 20; ++x) {
        mat[i][y][x] = 0;
      }
    }
  }
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    matrix_power(mat[i], row[i], col[i], a[i]);
  }
}
