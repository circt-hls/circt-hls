#include "memory_loop.h"
#ifndef AMOUNT_OF_TEST
#define AMOUNT_OF_TEST 1
#endif
int main(void) {
  int x[AMOUNT_OF_TEST][1000];
  int y[AMOUNT_OF_TEST][1000];
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    for (int j = 0; j < 1000; ++j) {
      x[i][j] = rand() % 100;
      y[i][j] = rand() % 100;
    }
  }
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    memory_loop(x[i], y[i]);
  }
}
