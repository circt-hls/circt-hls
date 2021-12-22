// RUN: hlstool --no_trace --rebuild --tb_file %s dynamic --run_sim

#include "loop_array.h"
#include <stdlib.h>

#ifndef N_KERNEL_CALLS
#define N_KERNEL_CALLS 1
#endif

int main(void) {
  int k[N_KERNEL_CALLS];
  int n[N_KERNEL_CALLS];
  int c[N_KERNEL_CALLS][10];
  srand(13);
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    k[i] = rand() % 10;
    n[i] = rand() % 10;
    for (int j = 0; j < 10; ++j) {
      c[i][j] = 0;
    }
  }
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    loop_array(n[i], k[i], c[i]);
  }
  return 0;
}
