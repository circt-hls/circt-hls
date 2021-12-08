// RUN: hlstool --no_trace --rebuild --tb_file %s dynamic --run_sim

#include "fir.h"
#include <stdlib.h>

#ifndef N_KERNEL_CALLS
#define N_KERNEL_CALLS 1
#endif

int main(void) {
  int d_i[N_KERNEL_CALLS][1000];
  int idx[N_KERNEL_CALLS][1000];
  int out[N_KERNEL_CALLS][1000];
  srand(13);
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    for (int j = 0; j < 1000; ++j) {
      d_i[i][j] = rand() % 100;
      idx[i][j] = rand() % 100;
    }
  }
  for (int i = 0; i < 1; ++i) {
    fir(d_i[i], idx[i]);
  }
}