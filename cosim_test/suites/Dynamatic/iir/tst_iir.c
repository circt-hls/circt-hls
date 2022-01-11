// RUN: hlstool --no_trace --rebuild --tb_file %s dynamic --run_sim

// hlstool will emit a {kernel_name}_tb_output.txt file when sim didn't crash.
// RUN: FileCheck --input-file *tb_output.txt %s

// CHECK: 0

#include "iir.h"
#include <stdlib.h>

#ifndef N_KERNEL_CALLS
#define N_KERNEL_CALLS 10
#endif

int main(void) {
  int y[N_KERNEL_CALLS][1000];
  int x[N_KERNEL_CALLS][1000];
  int b[N_KERNEL_CALLS];
  int a[N_KERNEL_CALLS];
  srand(13);
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    a[i] = rand();
    b[i] = rand();
    for (int j = 0; j < 1000; ++j) {
      y[i][j] = rand() % 1000;
      x[i][j] = rand() % 1000;
    }
  }
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    iir(y[i], x[i], a[i], b[i]);
  }
  return 0;
}
