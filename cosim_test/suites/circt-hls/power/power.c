#include "power.h"
int power(int a, int n) {
  int x = 1;
  for (int i = 0; i < n; ++i) {
    x *= a;
  }
  return x;
}
