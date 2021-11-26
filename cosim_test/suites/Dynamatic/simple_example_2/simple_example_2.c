// RUN: dyn_incrementally_lower %s

#include "simple_example_2.h"
void simple_example_2(int a[100]) {
  int x = 0;
  for (int i = 0; i < 100; ++i) {
    x++;
  }
  a[0] = x;
}
