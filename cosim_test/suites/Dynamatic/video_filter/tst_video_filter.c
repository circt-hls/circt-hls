// RUN: hlstool --no_trace --rebuild --tb_file %s dynamic --run_sim

// hlstool will emit a {kernel_name}_tb_output.txt file when sim didn't crash.
// RUN: FileCheck --input-file *tb_output.txt %s

// CHECK: 0

#include "video_filter.h"
#include <stdlib.h>

#ifndef N_KERNEL_CALLS
#define N_KERNEL_CALLS 1
#endif

int main(void) {
  int pixel_red[N_KERNEL_CALLS][30][30];
  int pixel_blue[N_KERNEL_CALLS][30][30];
  int pixel_green[N_KERNEL_CALLS][30][30];
  int offset[N_KERNEL_CALLS];
  int scale[N_KERNEL_CALLS];
  srand(13);
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    offset[i] = rand() % 100;
    scale[i] = rand() % 10;
    for (int y = 0; y < 30; ++y) {
      for (int x = 0; x < 30; ++x) {
        pixel_red[i][y][x] = rand() % 1000;
        pixel_blue[i][y][x] = rand() % 1000;
        pixel_green[i][y][x] = rand() % 1000;
      }
    }
  }
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    video_filter(pixel_red[i], pixel_blue[i], pixel_green[i], offset[i],
                 scale[i]);
  }
  return 0;
}
