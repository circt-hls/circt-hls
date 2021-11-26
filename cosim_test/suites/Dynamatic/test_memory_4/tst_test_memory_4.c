// RUN: dyn_hlt_lower %s

#include "test_memory_4.h"
#ifndef AMOUNT_OF_TEST
#define AMOUNT_OF_TEST 1
#endif
int main(void) {
  int a[AMOUNT_OF_TEST][5];
  int n[AMOUNT_OF_TEST];
  srand(13);
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    n[i] = 4;
    for (int j = 0; j < 5; ++j) {
      a[i][j] = rand() % 10;
    }
  }
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    test_memory_4(a[i], n[i]);
  }
}
