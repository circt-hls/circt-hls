#include "bicg.h"
int main(void) {
  int A[AMOUNT_OF_TEST][N][N];
  int s[AMOUNT_OF_TEST][N];
  int q[AMOUNT_OF_TEST][N];
  int p[AMOUNT_OF_TEST][N];
  int r[AMOUNT_OF_TEST][N];

  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    for (int y = 0; y < N; ++y) {
      s[i][y] = rand() % 100;
      q[i][y] = rand() % 100;
      p[i][y] = rand() % 100;
      r[i][y] = rand() % 100;
      for (int x = 0; x < N; ++x) {
        A[i][y][x] = rand() % 100;
      }
    }
  }

  // for(int i = 0; i < AMOUNT_OF_TEST; ++i){
  int i = 0;
  bicg(A[i], s[i], q[i], p[i], r[i]);
}
}
