// RUN: hlstool --tb_file %s dynamic

#include "vector_rescale.h"
#include <stdlib.h>

#ifndef AMOUNT_OF_TEST
#define AMOUNT_OF_TEST 1
#endif
int main(void) {
  int a[AMOUNT_OF_TEST][1000];
  int c[AMOUNT_OF_TEST];
  srand(13);
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    c[i] = rand() % 100;
    for (int j = 0; j < 1000; ++j) {
      a[i][j] = rand() % 100;
    }
  }
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    vector_rescale(a[i], c[i]);
  }
}
