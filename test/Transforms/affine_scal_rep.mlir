// RUN: hls-opt -split-input-file -affine-scalrep %s | FileCheck %s

// CHECK-LABEL: func.func @dead_reduction_buffer
// CHECK-NOT:     memref.alloca()
// CHECK-NOT:     affine.store
// CHECK:         %[[VAL:.+]] = affine.for
// CHECK-NOT:     affine.store
// CHECK:         return %[[VAL]] : i32
func.func @dead_reduction_buffer(%arg0: memref<64xi32>,
                            %arg1: memref<64xi32>) -> memref<i32> {
  %c0_i32 = arith.constant 0 : i32
  %0 = memref.alloca() : memref<i32>
  affine.store %c0_i32, %0[] : memref<i32>
  %1 = affine.for %arg2 = 0 to 64 iter_args(%arg3 = %c0_i32) -> (i32) {
    %2 = affine.load %arg0[%arg2] : memref<64xi32>
    %3 = affine.load %arg1[%arg2] : memref<64xi32>
    %4 = arith.muli %2, %3 : i32
    %5 = arith.addi %arg3, %4 : i32
    affine.yield %5 : i32
  }
  affine.store %1, %0[] : memref<i32>
  return %0 : memref<i32>
}
