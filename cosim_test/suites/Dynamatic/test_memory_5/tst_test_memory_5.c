// RUN: hlstool --no_trace --rebuild --tb_file %s dynamic --run_sim

#include "test_memory_5.h"
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
      a[i][j] = rand() % 10;
    }
  }
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    test_memory_5(a[i], n[i]);
  }
}
