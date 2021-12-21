// RUN: hls-opt --split-input-file --cosim-lower-compare %s | FileCheck %s

// CHECK-LABEL:   llvm.func @printf(!llvm.ptr<i8>, ...) -> i32
// CHECK:         llvm.mlir.global internal constant @cosimIntCmpErrStr("COSIM: %[[VAL_0:.*]] != %[[VAL_0]]")

// CHECK-LABEL:   func @compare_arith() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = arith.cmpi ne, %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           scf.if %[[VAL_2]] {
// CHECK:             %[[VAL_3:.*]] = llvm.mlir.addressof @cosimIntCmpErrStr : !llvm.ptr<array<15 x i8>>
// CHECK:             %[[VAL_4:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:             %[[VAL_5:.*]] = llvm.getelementptr %[[VAL_3]]{{\[}}%[[VAL_4]], %[[VAL_4]]] : (!llvm.ptr<array<15 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK:             %[[VAL_6:.*]] = llvm.call @printf(%[[VAL_5]], %[[VAL_0]], %[[VAL_1]]) : (!llvm.ptr<i8>, i32, i32) -> i32
// CHECK:           }
// CHECK:           return
// CHECK:         }
func @compare_arith() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    cosim.compare %c0_i32, %c1_i32 : i32
    return
}

// -----

// func @compare_memref() {
//     %0 = memref.alloca() : memref<100xi32>
//     %1 = memref.alloca() : memref<100xi32>
//     cosim.compare %0, %1 : i32
//     return
// }