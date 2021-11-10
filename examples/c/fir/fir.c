#define N 16

int32_t mul_without_mul(int32_t a, int32_t b)
{
  int32_t c = 0;
  int32_t i;
  for (i = 0; i < 32; i++)
  {
    if (b & 1)
      c += a;
    a <<= 1;
    b >>= 1;
  }
  return c;
}

void fir(int *y, int c[N], int x)
{
  static int shift_reg[N];
  int acc, data, i;
  acc = 0;
  for (i = N - 1; i >= 0; i--)
  {
    if (i == 0)
    {
      shift_reg[0] = x;
      data = x;
    }
    else
    {
      shift_reg[i] = shift_reg[i - 1];
      data = shift_reg[i];
    }
    acc += mul_without_mul(data, c[i]);
  }
  *y = acc;
}
