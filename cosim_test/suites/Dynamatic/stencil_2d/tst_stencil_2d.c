// RUN: hlt_test %s | run_test_scripts

#include "stencil_2d.h"
#ifndef AMOUNT_OF_TEST
#define AMOUNT_OF_TEST 1
#endif
int main(void) {
  int orig[AMOUNT_OF_TEST][900];
  int sol[AMOUNT_OF_TEST][900];
  int filter[AMOUNT_OF_TEST][10];
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    for (int j = 0; j < 900; ++j) {
      orig[i][j] = rand() % 100;
    }
    for (int j = 0; j < 10; ++j) {
      filter[i][j] = rand() % 100;
    }
  }
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    stencil_2d(orig[i], sol[i], filter[i]);
  }
}
