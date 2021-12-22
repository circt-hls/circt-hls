// RUN: hlstool --no_trace --rebuild --tb_file %s dynamic --run_sim

#include "if_loop_1.h"

#ifndef N_KERNEL_CALLS
#define N_KERNEL_CALLS 10
#endif

int main(void) {
  int a[N_KERNEL_CALLS][100];
  int n[N_KERNEL_CALLS];
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    n[i] = 100;
    for (int j = 0; j < 100; ++j) {
      a[i][j] = j % 10;
    }
  }
  for (int i = 0; i < 1; ++i) {
    if_loop_1(a[i], n[i]);
  }
}
