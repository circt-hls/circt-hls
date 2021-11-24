#include "sumi3_mem.h"
int main(void) {
  int a[AMOUNT_OF_TEST][1000];

  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    for (int j = 0; j < 1000; ++j) {
      a[i][j] = rand() % 10;
    }
  }

  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    int i = 0;
    sumi3_mem(a[i]);
  }
}
