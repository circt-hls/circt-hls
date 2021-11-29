// RUN: dyn_hlt_lower %s
// RUN: dyn_hlt_build_sim %s
// RUN: dyn_hlt_build_tb %s

#include "threshold.h"
#ifndef AMOUNT_OF_TEST
#define AMOUNT_OF_TEST 1
#endif
int main(void) {
  int red[AMOUNT_OF_TEST][1000];
  int green[AMOUNT_OF_TEST][1000];
  int blue[AMOUNT_OF_TEST][1000];
  int th[AMOUNT_OF_TEST];
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    th[i] = (rand() % 100);
    for (int j = 0; j < 1000; ++j) {
      red[i][j] = (rand() % 100);
      green[i][j] = (rand() % 100);
      blue[i][j] = (rand() % 100);
    }
  }
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    threshold(red[i], green[i], blue[i], th[i]);
  }
}