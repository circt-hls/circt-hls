#include "test_memory_3.h"
#ifndef AMOUNT_OF_TEST
#define AMOUNT_OF_TEST 1
#endif
void test_memory_3(int a[N], int n) {
  for (int i = 0; i < N; i++) {
    if (a[i] == n)
      a[i] = 0;
  }
}