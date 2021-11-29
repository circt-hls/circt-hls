// RUN: dyn_hlt_lower %s
// RUN: dyn_hlt_build_sim %s
// RUN: dyn_hlt_build_tb %s

#include "sumi3_mem.h"
#include <stdlib.h>

int main(void) {
  int a[AMOUNT_OF_TEST][1000];
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    for (int j = 0; j < 1000; ++j) {
      a[i][j] = rand() % 10;
    }
  }
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    sumi3_mem(a[i]);
  }
}
