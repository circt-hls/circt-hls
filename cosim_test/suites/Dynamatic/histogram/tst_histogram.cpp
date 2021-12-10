#include "histogram.h"
#include <stdlib.h>

#define AMOUNT_OF_TEST 1

int main(void) {
  int feature[AMOUNT_OF_TEST][N];
  int weight[AMOUNT_OF_TEST][N];
  int hist[AMOUNT_OF_TEST][N];
  int n[AMOUNT_OF_TEST];

  srand(13);
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    n[i] = N;
    for (int j = 0; j < N; ++j) {
      feature[i][j] = rand() % N;
      weight[i][j] = rand() % 100;
      hist[i][j] = rand() % 100;
    }
  }

  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    histogram(feature[i], weight[i], hist[i], n[i]);
  }
}
