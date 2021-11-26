// RUN: dyn_incrementally_lower %s

#include "loop_array.h"
void loop_array(int n, int k, int c[10]) {
  for (int i = 1; i < n; i++) {
    c[i] = k + c[i - 1];
  }
}