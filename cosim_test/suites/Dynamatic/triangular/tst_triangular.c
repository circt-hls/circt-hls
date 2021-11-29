// RUN: dyn_hlt_lower %s
// RUN: dyn_hlt_build_sim %s

#include "triangular.h"
#ifndef AMOUNT_OF_TEST
#define AMOUNT_OF_TEST 1
#endif
int main(void) {
  int xArray[AMOUNT_OF_TEST][10];
  int A[AMOUNT_OF_TEST][10][10];
  int n[AMOUNT_OF_TEST];
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    n[i] = 10; //(rand() % 100);
    for (int x = 0; x < 10; ++x) {
      xArray[i][x] = rand() % 100;
      for (int y = 0; y < 10; ++y) {
        A[i][y][x] = rand() % 100;
      }
    }
  }
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    triangular(xArray[i], A[i], n[i]);
  }
}
