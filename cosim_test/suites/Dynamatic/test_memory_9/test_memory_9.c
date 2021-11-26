// RUN: dyn_incrementally_lower %s

#include "test_memory_9.h"
void test_memory_9(int a[4], int n) {
  int x;
  for (int i = 1; i < n; i++) {
    a[i] = 5 * a[i - 1];
  }
}
