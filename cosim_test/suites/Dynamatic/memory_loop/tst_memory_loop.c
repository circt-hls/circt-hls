// RUN: hlstool --no_trace --rebuild --tb_file %s dynamic --run_sim

#include "memory_loop.h"
#ifndef N_KERNEL_CALLS
#define N_KERNEL_CALLS 1
#endif

int main(void) {
  int x[N_KERNEL_CALLS][1000];
  int y[N_KERNEL_CALLS][1000];
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    for (int j = 0; j < 1000; ++j) {
      x[i][j] = rand() % 100;
      y[i][j] = rand() % 100;
    }
  }
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    memory_loop(x[i], y[i]);
  }
}
