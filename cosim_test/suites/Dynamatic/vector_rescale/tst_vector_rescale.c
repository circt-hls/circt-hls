// RUN: hlstool --no_trace --rebuild --tb_file %s dynamic --run_sim

// hlstool will emit a {kernel_name}_tb_output.txt file when sim didn't crash.
// RUN: FileCheck --input-file *tb_output.txt %s

// CHECK: 0

#include "vector_rescale.h"
#include <stdlib.h>

#ifndef N_KERNEL_CALLS
#define N_KERNEL_CALLS 1
#endif

int main(void) {
  int a[N_KERNEL_CALLS][1000];
  int c[N_KERNEL_CALLS];
  srand(13);
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    c[i] = rand() % 100;
    for (int j = 0; j < 1000; ++j) {
      a[i][j] = rand() % 100;
    }
  }
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    vector_rescale(a[i], c[i]);
  }
  return 0;
}
