#include "if_loop_3.h"
int if_loop_3(int a[100], int b[100], int n) {
  int i;
  int dist;
  int sum = 1000;

  for (i = 0; i < n; i++) {
    dist = a[i] - b[i];

    if (dist >= 0) {

      sum = (sum / dist);
    }
  }
  return sum;
}