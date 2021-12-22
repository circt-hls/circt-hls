// RUN: hlstool --no_trace --rebuild --tb_file %s dynamic --run_sim

#include "sumi3_mem.h"
#include <stdlib.h>

#define N_KERNEL_CALLS 2

int main(void) {
  int a[N_KERNEL_CALLS][1000];
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    for (int j = 0; j < 1000; ++j) {
      a[i][j] = rand() % 10;
    }
  }
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    sumi3_mem(a[i]);
  }
  return 0;
}
