// RUN: dyn_hlt_lower %s
// RUN: dyn_hlt_build_sim %s
// RUN: dyn_hlt_build_tb %s

#include "loop_array.h"
#include <stdlib.h>

#ifndef AMOUNT_OF_TEST
#define AMOUNT_OF_TEST 1
#endif
int main(void) {
  int k[AMOUNT_OF_TEST];
  int n[AMOUNT_OF_TEST];
  int c[AMOUNT_OF_TEST][10];
  srand(13);
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    k[i] = rand() % 10;
    n[i] = rand() % 10;
    for (int j = 0; j < 10; ++j) {
      c[i][j] = 0;
    }
  }
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    loop_array(n[i], k[i], c[i]);
  }
}
