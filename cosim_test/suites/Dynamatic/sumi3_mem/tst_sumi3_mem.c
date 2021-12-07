// RUN: hlstool --tb_file %s dynamic

#include "sumi3_mem.h"
#include <stdlib.h>

#define AMOUNT_OF_TEST 2

int main(void) {
  int a[AMOUNT_OF_TEST][1000];
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    for (int j = 0; j < 1000; ++j) {
      a[i][j] = rand() % 10;
    }
  }
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    sumi3_mem(a[i]);
  }
}
