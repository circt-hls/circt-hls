// RUN: hlstool --no_trace --rebuild --tb_file %s dynamic --run_sim

#include "test_memory_3.h"
#include <stdlib.h>

#ifndef N_KERNEL_CALLS
#define N_KERNEL_CALLS 1
#endif

int main(void) {
  int a[N_KERNEL_CALLS][N];
  int n[N_KERNEL_CALLS];
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    n[i] = N / 2;
    for (int j = 0; j < N; ++j) {
      a[i][j] = j;
    }
  }
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    test_memory_3(a[i], n[i]);
  }
  return 0;
}
