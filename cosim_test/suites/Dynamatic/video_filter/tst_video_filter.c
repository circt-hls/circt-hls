// RUN: hlstool --rebuild --tb_file %s dynamic

#include "video_filter.h"
#include <stdlib.h>

#ifndef AMOUNT_OF_TEST
#define AMOUNT_OF_TEST 1
#endif
int main(void) {
  int pixel_red[AMOUNT_OF_TEST][30][30];
  int pixel_blue[AMOUNT_OF_TEST][30][30];
  int pixel_green[AMOUNT_OF_TEST][30][30];
  int offset[AMOUNT_OF_TEST];
  int scale[AMOUNT_OF_TEST];
  srand(13);
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
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
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    video_filter(pixel_red[i], pixel_blue[i], pixel_green[i], offset[i],
                 scale[i]);
  }
}
