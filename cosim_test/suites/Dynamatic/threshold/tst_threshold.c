// RUN: hlstool --no_trace --rebuild --tb_file %s dynamic --run_sim

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