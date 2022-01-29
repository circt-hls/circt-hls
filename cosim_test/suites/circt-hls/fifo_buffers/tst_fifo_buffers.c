// RUN: hlstool --rebuild --tb_file %s --mlir_kernel dynamic-polygeist --run_sim --hs_kernel
// CHECK: 0
#ifndef N_KERNEL_CALLS
#define N_KERNEL_CALLS 10
#endif

int fifo_buffers(int, int);
int main(void) {
  int a[N_KERNEL_CALLS];
  int b[N_KERNEL_CALLS];
  int c[N_KERNEL_CALLS];
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    a[i] = 2 * i;
    b[i] = -i;
  }
  for (int i = 0; i < N_KERNEL_CALLS; ++i) {
    c[i] = fifo_buffers(a[i], b[i]);
  }
}
