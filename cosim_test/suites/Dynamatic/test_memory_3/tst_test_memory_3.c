// RUN: dyn_hlt_lower %s
// RUN: dyn_hlt_build_sim %s

#include "test_memory_3.h"
int main(void) {
  int a[AMOUNT_OF_TEST][4];
  int n[AMOUNT_OF_TEST];
  srand(13);
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    n[i] = 4;
    for (int j = 0; j < 4; ++j) {
      a[i][j] = rand();
    }
  }
  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    test_memory_3(a[i], n[i]);
  }
}
