// RUN: dyn_incrementally_lower %s

#include "mul_example.h"

void mul_example(int a[100]) {
  for (int i = 0; i < 100; ++i) {
    int x = a[i];
    a[i] = x * x * x;
  }
}