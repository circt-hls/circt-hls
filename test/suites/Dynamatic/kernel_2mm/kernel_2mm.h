#pragma once
#define NI 10
#define NJ 10
#define NK 10
#define NL 10
#define N 10

void kernel_2mm(int alpha, int beta, int tmp[N][N], int A[N][N], int B[N][N],
                int C[N][N], int D[N][N]);
