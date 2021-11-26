#include "test_memory_3.h"
#ifndef AMOUNT_OF_TEST
#define AMOUNT_OF_TEST 1
#endif
void test_memory_3(int a[4], int n) {
  for (int i = 0; i < n; i++) {
    if (a[i] == n)
      a[i] = 0;
  }
}