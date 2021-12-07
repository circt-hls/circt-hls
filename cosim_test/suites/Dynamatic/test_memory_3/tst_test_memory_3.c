// RUN: hlt_test %s | run_test_scripts

#include "test_memory_3.h"
#include <stdlib.h>

#define AMOUNT_OF_TEST 1

int main(void) {
  int a[AMOUNT_OF_TEST][N];
  int n[AMOUNT_OF_TEST];
  srand(13);
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    n[i] = N / 2;
    for (int j = 0; j < N; ++j) {
      a[i][j] = j;
    }
  }
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    test_memory_3(a[i], n[i]);
  }
}
