// RUN: dyn_incrementally_lower %s

#include "if_loop_1.h"
int if_loop_1(int a[100], int n) {
  int i;
  int tmp;
  int c = 5;
  int sum = 0;
  for (i = 0; i < n; i++) {
    tmp = a[i] * c;
    if (tmp > 10) {
      sum = tmp + sum;
    }
  }
  return sum;
}