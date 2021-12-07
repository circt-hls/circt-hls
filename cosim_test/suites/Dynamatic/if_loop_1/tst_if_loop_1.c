// RUN: hlstool --rebuild --tb_file %s dynamic

#include "if_loop_1.h"

#ifndef AMOUNT_OF_TEST
#define AMOUNT_OF_TEST 1
#endif
int main(void) {
  int a[AMOUNT_OF_TEST][100];
  int n[AMOUNT_OF_TEST];
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    n[i] = 100;
    for (int j = 0; j < 100; ++j) {
      a[i][j] = rand() % 100;
    }
  }
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    if_loop_1(a[i], n[i]);
  }
}
