#include "pivot.h"
void pivot(int x[1000], int a[1000], int n, int k) {
  int i;
  for (i = k + 1; i <= n; ++i) {
    x[k] = x[k] - a[i] * x[i];
  }
}