// RUN: dyn_incrementally_lower %s

int exponent(int arg0, int arg1) {
  int acc = arg0;
  while (arg1 != 0) {
    acc += arg0;
    arg1--;
  }
  return acc;
}