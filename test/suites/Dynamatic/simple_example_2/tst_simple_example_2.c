#include "simple_example_2.h"

#define AMOUNT_OF_TEST 1

int main(void) {
  int a[AMOUNT_OF_TEST][100];
  int b[AMOUNT_OF_TEST][100];
  int c[AMOUNT_OF_TEST];

  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    c[i] = 3;
    for (int j = 0; j < 100; ++j) {
      a[i][j] = j;
      b[i][j] = 99 - j;
    }
  }

  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    simple_example_2(a[0]);
  }
}