#include "image_resize.h"

#ifndef AMOUNT_OF_TEST
#define AMOUNT_OF_TEST 1
#endif

int main(void) {
  int a[AMOUNT_OF_TEST][30][30];
  int c[AMOUNT_OF_TEST];

  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    c[i] = 1000;
    for (int y = 0; y < 30; ++y) {
      for (int x = 0; x < 30; ++x) {
        a[i][y][x] = rand() % 100;
      }
    }
  }

  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    image_resize(a[i], c[i]);
  }
}
