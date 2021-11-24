#include "matrix_power.h"
#define AMOUNT_OF_TEST 1

int main(void) {
  int mat[AMOUNT_OF_TEST][20][20];
  int row[AMOUNT_OF_TEST][20];
  int col[AMOUNT_OF_TEST][20];
  int a[AMOUNT_OF_TEST][20];

  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    for (int y = 0; y < 20; ++y) {
      col[i][y] = rand() % 20;
      row[i][y] = rand() % 20;
      a[i][y] = rand();
      for (int x = 0; x < 20; ++x) {
        mat[i][y][x] = 0;
      }
    }
  }

  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    int i = 0;
    matrix_power(mat[i], row[i], col[i], a[i]);
  }
}
