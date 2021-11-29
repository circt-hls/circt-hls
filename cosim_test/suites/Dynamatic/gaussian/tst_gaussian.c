// RUN: dyn_hlt_lower %s
// RUN: dyn_hlt_build_sim %s
// RUN: dyn_hlt_build_tb %s

#include "gaussian.h"
#include <stdlib.h>

#ifndef AMOUNT_OF_TEST
#define AMOUNT_OF_TEST 1
#endif
int main(void) {
  int c[AMOUNT_OF_TEST][20];
  int A[AMOUNT_OF_TEST][20][20];
  srand(13);
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    for (int y = 0; y < 20; ++y) {
      c[i][y] = 1; // rand()%20;
      for (int x = 0; x < 20; ++x) {
        A[i][y][x] = 1; // rand()%20;
      }
    }
  }
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    int i = 0;
    gaussian(c[i], A[i]);
  }
}
