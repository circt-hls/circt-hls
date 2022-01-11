// RUN: hlstool --no_trace --rebuild --tb_file %s dynamic --run_sim

// hlstool will emit a {kernel_name}_tb_output.txt file when sim didn't crash.
// RUN: FileCheck --input-file *tb_output.txt %s

// CHECK: 0

#include "matvec.h"
#ifndef N_KERNEL_CALLS
#define N_KERNEL_CALLS 10
#endif

int main(void) {
  int M[N_KERNEL_CALLS][30][30];
  int V[N_KERNEL_CALLS][30];
  int Out[N_KERNEL_CALLS][30];
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    for (int y = 0; y < 30; ++y) {
      V[i][y] = rand() % 100;
      for (int x = 0; x < 30; ++x) {
        M[i][y][x] = rand() % 100;
      }
    }
  }
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    matvec(M[i], V[i], Out[i]);
  }
  return 0;
}
