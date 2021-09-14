#define N 16

void fir(int *y, int c[N], int x) {
  static int shift_reg[N];
  int acc, data, i;
  acc = 0;
  for (i = N - 1; i >= 0; i--) {
    if (i == 0) {
      shift_reg[0] = x;
      data = x;
    } else {
      shift_reg[i] = shift_reg[i - 1];
      data = shift_reg[i];
    }
    acc += data * c[i];
  }
  *y = acc;
}
