#include "histogram.h"

void histogram(int feature[N], int weight[N], int hist[N], int n) {
  for (int i = 0; i < n; ++i) {
    int m = feature[i];
    int wt = weight[i];
    int x = hist[m];
    hist[m] = x + wt;
  }
}
