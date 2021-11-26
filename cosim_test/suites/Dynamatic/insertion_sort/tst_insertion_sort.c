#include "insertion_sort.h"

#ifndef AMOUNT_OF_TEST
#define AMOUNT_OF_TEST 1
#endif

int main(void) {
  int a[AMOUNT_OF_TEST][1000];
  int n[AMOUNT_OF_TEST];

  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    n[i] = 30;
    for (int j = 0; j < 1000; ++j) {
      a[i][j] = rand() % 10;
    }
  }

  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    insertion_sort(a[i], n[i]);
  }
}
