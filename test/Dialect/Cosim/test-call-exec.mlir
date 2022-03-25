// RUN: hls-opt --split-input-file --cosim-lower-compare %s > cosim_compare.mlir
// RUN: mlir-opt --convert-scf-to-cf --convert-cf-to-llvm --convert-func-to-llvm cosim_compare.mlir > cosim_compare_llvm.mlir
// RUN: mlir-cpu-runner \
// RUN:    -e main -entry-point-result=i32 -O3                    \
// RUN:    -shared-libs=%llvm_shlibdir/libmlir_c_runner_utils.so  \
// RUN:    -shared-libs=%llvm_shlibdir/libmlir_runner_utils.so    \
// RUN:    cosim_compare_llvm.mlir | FileCheck %s

// CHECK: COSIM: 0 != 1
func @main() -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    cosim.compare %c0_i32, %c1_i32 : i32
    return %c0_i32 : i32
}
