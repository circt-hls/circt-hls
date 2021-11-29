// RUN: dyn_hlt_lower %s
// RUN: dyn_hlt_build_sim %s
// RUN: dyn_hlt_build_tb %s

#include "if_loop_3.h"

#ifndef AMOUNT_OF_TEST
#define AMOUNT_OF_TEST 1
#endif
int main(void) {
  int a[AMOUNT_OF_TEST][100];
  int b[AMOUNT_OF_TEST][100];
  int n[AMOUNT_OF_TEST];
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    n[i] = 100;
    for (int j = 0; j < 100; ++j) {
      a[i][j] = rand() % 10;
      b[i][j] = a[i][j] + 1;
    }
  }
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    int i = 0;
    if_loop_3(a[i], b[i], n[i]);
  }
}
