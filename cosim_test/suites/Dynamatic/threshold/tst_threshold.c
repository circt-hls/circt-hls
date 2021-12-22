// RUN: hlstool --no_trace --rebuild --tb_file %s dynamic --run_sim

#include "threshold.h"
#ifndef N_KERNEL_CALLS
#define N_KERNEL_CALLS 1
#endif

int main(void) {
  int red[N_KERNEL_CALLS][1000];
  int green[N_KERNEL_CALLS][1000];
  int blue[N_KERNEL_CALLS][1000];
  int th[N_KERNEL_CALLS];
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    th[i] = (rand() % 100);
    for (int j = 0; j < 1000; ++j) {
      red[i][j] = (rand() % 100);
      green[i][j] = (rand() % 100);
      blue[i][j] = (rand() % 100);
    }
  }
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    threshold(red[i], green[i], blue[i], th[i]);
  }
  return 0;
}