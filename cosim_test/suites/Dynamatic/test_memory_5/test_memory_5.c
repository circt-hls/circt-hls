#include "test_memory_5.h"
void test_memory_5(int a[4], int n) {
  int x;
  for (int i = 2; i < n; i++) {
    a[i] = a[i - 1] + a[i - 2] + 5;
  }
}