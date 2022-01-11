// RUN: hlstool --no_trace --rebuild --tb_file %s dynamic-polygeist --run_sim

// hlstool will emit a {kernel_name}_tb_output.txt file when sim didn't crash.
// RUN: FileCheck --input-file *tb_output.txt %s

// CHECK: 0

#include "test_memory_2.h"
#include <stdlib.h>

#ifndef N_KERNEL_CALLS
#define N_KERNEL_CALLS 10
#endif

int main(void) {
  int a[N_KERNEL_CALLS][5];
  int n[N_KERNEL_CALLS];
  srand(13);
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    n[i] = 4;
    for (int j = 0; j < 5; ++j) {
      a[i][j] = rand() % 10;
    }
  }
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    test_memory_2(a[i], n[i]);
  }
  return 0;
}
