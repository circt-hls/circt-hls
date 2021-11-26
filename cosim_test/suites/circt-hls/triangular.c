#define N 8
int triangular(int i) {
  int acc = 0;
  int arr[N];
  for (int j = 0; j < i; ++j)
    arr[j] = j;
  for (int j = 0; j < i; ++j)
    acc += arr[j];
  return acc;
}
