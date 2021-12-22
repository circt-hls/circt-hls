// RUN: hlstool --no_trace --rebuild --tb_file %s dynamic --run_sim

#include "image_resize.h"

#ifndef N_KERNEL_CALLS
#define N_KERNEL_CALLS 2
#endif

int main(void) {
  int a[N_KERNEL_CALLS][30][30];
  int c[N_KERNEL_CALLS];
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    c[i] = 1000;
    for (int y = 0; y < 30; ++y) {
      for (int x = 0; x < 30; ++x) {
        a[i][y][x] = rand() % 100;
      }
    }
  }
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    image_resize(a[i], c[i]);
  }
  return 0;
}
