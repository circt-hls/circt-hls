// RUN: dyn_hlt_lower %s
// RUN: dyn_hlt_build_sim %s
// RUN: dyn_hlt_build_tb %s

#include "test_memory_8.h"
#include <stdlib.h>

#ifndef AMOUNT_OF_TEST
#define AMOUNT_OF_TEST 1
#endif
int main(void) {
  int a[AMOUNT_OF_TEST][4];
  int n[AMOUNT_OF_TEST];
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    n[i] = 4;
    for (int j = 0; j < 4; ++j) {
      a[i][j] = (rand() % 100);
    }
  }
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    test_memory_8(a[i], n[i]);
  }
}
