#include "fir.h"

int main(void) {
  int d_i[AMOUNT_OF_TEST][1000];
  int idx[AMOUNT_OF_TEST][1000];
  int out[AMOUNT_OF_TEST][1000];

  srand(13);
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    for (int j = 0; j < 1000; ++j) {
      d_i[0][j] = rand() % 100;
      idx[0][j] = rand() % 100;
    }
  }

  for (int i = 0; i < 1; ++i) {
    fir(d_i[0], idx[0]);
  }
}