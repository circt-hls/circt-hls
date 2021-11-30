#include "test_memory_2.h"

void test_memory_2(int a[5], int n) {
  int x;
  for (int i = 0; i < n; i++) {
    a[i] = a[i] + 5;
  }
}
