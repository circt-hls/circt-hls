// RUN: hlstool --no_trace --rebuild --tb_file %s dynamic-polygeist --run_sim

// hlstool will emit a {kernel_name}_tb_output.txt file when sim didn't crash.
// RUN: FileCheck --input-file *tb_output.txt %s

// CHECK: 0

#include "memory_loop.h"
#ifndef N_KERNEL_CALLS
#define N_KERNEL_CALLS 10
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
  return 0;
}
