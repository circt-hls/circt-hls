#include "simple_example_1.h"
int simple_example_1(int a[100]) {
  for (int i = 0; i < 100; ++i) {
    a[i] = i;
  }
  return 0;
}
