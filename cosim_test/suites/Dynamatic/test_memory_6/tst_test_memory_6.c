// RUN: hlstool --no_trace --rebuild --tb_file %s dynamic --run_sim

#include "test_memory_6.h"
#include <stdlib.h>

#ifndef N_KERNEL_CALLS
#define N_KERNEL_CALLS 1
#endif

int main(void) {
  int a[N_KERNEL_CALLS][4];
  int n[N_KERNEL_CALLS];
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    n[i] = 4;
    for (int j = 0; j < 4; ++j) {
      a[i][j] = rand() % 10;
    }
  }
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    test_memory_6(a[i], n[i]);
  }
}
