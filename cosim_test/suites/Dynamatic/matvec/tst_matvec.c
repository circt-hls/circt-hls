// RUN: hlstool --no_trace --rebuild --tb_file %s dynamic --run_sim

#include "matvec.h"
#ifndef AMOUNT_OF_TEST
#define AMOUNT_OF_TEST 1
#endif
int main(void) {
  int M[AMOUNT_OF_TEST][30][30];
  int V[AMOUNT_OF_TEST][30];
  int Out[AMOUNT_OF_TEST][30];
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    for (int y = 0; y < 30; ++y) {
      V[i][y] = rand() % 100;
      for (int x = 0; x < 30; ++x) {
        M[i][y][x] = rand() % 100;
      }
    }
  }
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    matvec(M[i], V[i], Out[i]);
  }
}
