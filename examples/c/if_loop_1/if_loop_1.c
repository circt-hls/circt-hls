typedef int in_int_t;

int if_loop_1(in_int_t a[100], in_int_t n) {
  int i;
  int tmp;
  int c = 5;
  int sum = 0;
  for (i = 0; i < n; i++) {
    tmp = a[i] * c;
    if (tmp > 10)
      sum = tmp + sum;
  }
  return sum;
}
