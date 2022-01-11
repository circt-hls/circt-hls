// RUN: hlstool --no_trace --rebuild --tb_file %s dynamic-polygeist --run_sim

// hlstool will emit a {kernel_name}_tb_output.txt file when sim didn't crash.
// RUN: FileCheck --input-file *tb_output.txt %s

// CHECK: 0

#include "sumi3_mem.h"
#include <stdlib.h>

#define N_KERNEL_CALLS 10

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
