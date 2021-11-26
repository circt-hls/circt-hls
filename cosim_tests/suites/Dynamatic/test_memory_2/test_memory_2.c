#include "test_memory_2.h"
void test_memory_1(int a[4], int n) {
  int x;
  for (int i = 0; i < n; i++) {
    a[i] = a[i] + 5;
  }
}
