#include "fir.h"
int fir(int d_i[1000], int idx[1000]) {
  int i;
  int tmp = 0;
For_Loop:
  for (i = 0; i < 1000; i++) {
    tmp += idx[i] * d_i[999 - i];
  }
  return tmp;
}