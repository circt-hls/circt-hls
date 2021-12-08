// RUN: hlstool --no_trace --rebuild --tb_file %s dynamic --run_sim

#include "mul_example.h"
#ifndef N_KERNEL_CALLS
#define N_KERNEL_CALLS 1
#endif

int main(void) {
  int a[N_KERNEL_CALLS][100];
  int b[N_KERNEL_CALLS][100];
  int c[N_KERNEL_CALLS];
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    c[i] = 3;
    for (int j = 0; j < 100; ++j) {
      a[i][j] = j;
      b[i][j] = 99 - j;
    }
  }
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    mul_example(a[i]);
  }
}