#include "test_memory_4.h"
void test_memory_4(int a[4], int n) {
  int x = 0;
  for (int i = 0; i < n - 1; i++) {
    a[i] = x;
    x = a[i + 1];
  }
}
