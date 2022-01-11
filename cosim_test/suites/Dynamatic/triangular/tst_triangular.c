// RUN: hlstool --no_trace --rebuild --tb_file %s dynamic --run_sim

// hlstool will emit a {kernel_name}_tb_output.txt file when sim didn't crash.
// RUN: FileCheck --input-file *tb_output.txt %s

// CHECK: 0

#include "triangular.h"
#ifndef N_KERNEL_CALLS
#define N_KERNEL_CALLS 10
#endif

int main(void) {
  int xArray[N_KERNEL_CALLS][10];
  int A[N_KERNEL_CALLS][10][10];
  int n[N_KERNEL_CALLS];
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    n[i] = 10; //(rand() % 100);
    for (int x = 0; x < 10; ++x) {
      xArray[i][x] = rand() % 100;
      for (int y = 0; y < 10; ++y) {
        A[i][y][x] = rand() % 100;
      }
    }
  }
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    triangular(xArray[i], A[i], n[i]);
  }
  return 0;
}
