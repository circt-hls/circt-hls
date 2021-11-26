// RUN: dyn_hlt_lower %s

#include "matrix.h"
#ifndef AMOUNT_OF_TEST
#define AMOUNT_OF_TEST 1
#endif
int main(void) {
  int in_a[AMOUNT_OF_TEST][A_ROWS][B_ROWS];
  int in_b[AMOUNT_OF_TEST][A_ROWS][B_ROWS];
  int out_c[AMOUNT_OF_TEST][A_ROWS][B_ROWS];
  srand(13);
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    for (int y = 0; y < A_ROWS; ++y) {
      for (int x = 0; x < A_ROWS; ++x) {
        in_a[i][y][x] = rand() % 10;
        in_b[i][y][x] = rand() % 10;
      }
    }
  }
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    matrix(in_a[i], in_b[i], out_c[i]);
  }
}
