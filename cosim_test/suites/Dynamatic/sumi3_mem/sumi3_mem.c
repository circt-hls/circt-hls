// RUN: dyn_incrementally_lower %s

#include "sumi3_mem.h"
int sumi3_mem(int a[1000]) {
  int sum = 0;
  for (int i = 0; i < 1000; i++) {
    int x = a[i];
    sum += x * x * x;
  }
  return sum;
}