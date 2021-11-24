#include "loop_array.h"

#define AMOUNT_OF_TEST 1

int main(void) {
  int k[AMOUNT_OF_TEST];
  int n[AMOUNT_OF_TEST];
  int c[AMOUNT_OF_TEST][10];

  srand(13);
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    k[i] = rand() % 10;
    n[i] = rand() % 10;
    for (int j = 0; j < 10; ++j) {
      c[i][j] = 0;
    }
  }

  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    int i = 0;
    loop_array(n[i], k[i], c[i]);
  }
}
