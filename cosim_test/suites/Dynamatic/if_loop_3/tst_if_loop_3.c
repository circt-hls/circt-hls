// RUN: hlstool --no_trace --rebuild --tb_file %s dynamic --run_sim

#include "if_loop_3.h"

#ifndef N_KERNEL_CALLS
#define N_KERNEL_CALLS 1
#endif

int main(void) {
  int a[N_KERNEL_CALLS][100];
  int b[N_KERNEL_CALLS][100];
  int n[N_KERNEL_CALLS];
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    n[i] = 100;
    for (int j = 0; j < 100; ++j) {
      a[i][j] = rand() % 10;
      b[i][j] = a[i][j] + 1;
    }
  }
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    if_loop_3(a[i], b[i], n[i]);
  }
  return 0;
}
