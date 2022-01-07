#include "simple_example_1.h"
int simple_example_3(int a) {
  int x = 0;
  for (int i = 0; i < 100; ++i) {
    x += i;
  }
  return x;
}
