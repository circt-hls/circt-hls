// RUN: dyn_hlt_lower %s
// RUN: dyn_hlt_build_sim %s
// RUN: dyn_hlt_build_tb %s

#include "pivot.h"
#ifndef AMOUNT_OF_TEST
#define AMOUNT_OF_TEST 1
#endif
int main(void) {
  int x[AMOUNT_OF_TEST][1000];
  int a[AMOUNT_OF_TEST][1000];
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    for (int j = 0; j < 1000; ++j) {
      x[i][j] = rand() % 100;
      a[i][j] = rand() % 100;
    }
  }
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    pivot(x[i], a[i], 100, 2);
  }
}