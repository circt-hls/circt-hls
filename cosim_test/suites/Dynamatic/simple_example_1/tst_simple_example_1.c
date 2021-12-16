// RUN: hlstool --no_trace --rebuild --tb_file %s dynamic --run_sim

#include "simple_example_1.h"
#ifndef N_KERNEL_CALLS
#define N_KERNEL_CALLS 1
#endif

int main(void) {
  int a[N_KERNEL_CALLS][100];
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    for (int j = 0; j < 100; ++j) {
      a[i][j] = j;
    }
  }
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    simple_example_1(a[i]);
  }
}