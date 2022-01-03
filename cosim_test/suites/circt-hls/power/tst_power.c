// RUN: hlstool --no_trace --rebuild --tb_file %s dynamic --run_sim

// hlstool will emit a {kernel_name}_tb_output.txt file when sim didn't crash.
// RUN: FileCheck --input-file *tb_output.txt %s

// CHECK: 0

#include "power.h"
#ifndef N_KERNEL_CALLS
#define N_KERNEL_CALLS 10
#endif

int main(void) {
  for (int i = 0; i < N_KERNEL_CALLS; ++i)
    power(2, i);
  return 0;
}