// RUN: hlstool --no_trace --rebuild --tb_file %s dynamic-polygeist --run_sim

// hlstool will emit a {kernel_name}_tb_output.txt file when sim didn't crash.
// RUN: FileCheck --input-file *tb_output.txt %s

// CHECK: 0

#include "stencil_2d.h"
#ifndef N_KERNEL_CALLS
#define N_KERNEL_CALLS 10
#endif

int main(void) {
  int orig[N_KERNEL_CALLS][900];
  int sol[N_KERNEL_CALLS][900];
  int filter[N_KERNEL_CALLS][10];
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    for (int j = 0; j < 900; ++j) {
      orig[i][j] = rand() % 100;
    }
    for (int j = 0; j < 10; ++j) {
      filter[i][j] = rand() % 100;
    }
  }
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    stencil_2d(orig[i], sol[i], filter[i]);
  }
  return 0;
}
