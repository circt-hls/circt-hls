// RUN: hls-opt --split-input-file --cosim-lower-compare %s | FileCheck %s

// CHECK-LABEL:   func @compare_memref() {
// CHECK:           %[[VAL_0:.*]] = memref.alloca() : memref<100xi32>
// CHECK:           %[[VAL_1:.*]] = memref.alloca() : memref<100xi32>
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 100 : index
// CHECK:           scf.for %[[VAL_5:.*]] = %[[VAL_2]] to %[[VAL_4]] step %[[VAL_3]] {
// CHECK:             %[[VAL_6:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_5]]] : memref<100xi32>
// CHECK:             %[[VAL_7:.*]] = memref.load %[[VAL_1]]{{\[}}%[[VAL_5]]] : memref<100xi32>
// CHECK:             %[[VAL_8:.*]] = arith.cmpi ne, %[[VAL_6]], %[[VAL_7]] : i32
// CHECK:             scf.if %[[VAL_8]] {
// CHECK:               %[[VAL_9:.*]] = llvm.mlir.addressof @cosimIntCmpErrStr : !llvm.ptr<array<15 x i8>>
// CHECK:               %[[VAL_10:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:               %[[VAL_11:.*]] = llvm.getelementptr %[[VAL_9]]{{\[}}%[[VAL_10]], %[[VAL_10]]] : (!llvm.ptr<array<15 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK:               %[[VAL_12:.*]] = llvm.call @printf(%[[VAL_11]], %[[VAL_6]], %[[VAL_7]]) : (!llvm.ptr<i8>, i32, i32) -> i32
// CHECK:             }
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

// CHECK-LABEL:   func @compare_multidim_memref() {
// CHECK:           %[[VAL_0:.*]] = memref.alloca() : memref<100x64xi32>
// CHECK:           %[[VAL_1:.*]] = memref.alloca() : memref<100x64xi32>
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 100 : index
// CHECK:           scf.for %[[VAL_5:.*]] = %[[VAL_2]] to %[[VAL_4]] step %[[VAL_3]] {
// CHECK:             %[[VAL_6:.*]] = arith.constant 64 : index
// CHECK:             scf.for %[[VAL_7:.*]] = %[[VAL_2]] to %[[VAL_6]] step %[[VAL_3]] {
// CHECK:               %[[VAL_8:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_5]], %[[VAL_7]]] : memref<100x64xi32>
// CHECK:               %[[VAL_9:.*]] = memref.load %[[VAL_1]]{{\[}}%[[VAL_5]], %[[VAL_7]]] : memref<100x64xi32>
// CHECK:               %[[VAL_10:.*]] = arith.cmpi ne, %[[VAL_8]], %[[VAL_9]] : i32
// CHECK:               scf.if %[[VAL_10]] {
// CHECK:                 %[[VAL_11:.*]] = llvm.mlir.addressof @cosimIntCmpErrStr : !llvm.ptr<array<15 x i8>>
// CHECK:                 %[[VAL_12:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:                 %[[VAL_13:.*]] = llvm.getelementptr %[[VAL_11]]{{\[}}%[[VAL_12]], %[[VAL_12]]] : (!llvm.ptr<array<15 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK:                 %[[VAL_14:.*]] = llvm.call @printf(%[[VAL_13]], %[[VAL_8]], %[[VAL_9]]) : (!llvm.ptr<i8>, i32, i32) -> i32
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
func @compare_memref() {
    %0 = memref.alloca() : memref<100xi32>
    %1 = memref.alloca() : memref<100xi32>
    cosim.compare %0, %1 : memref<100xi32>
    return
}

// -----

func @compare_multidim_memref() {
    %0 = memref.alloca() : memref<100x64xi32>
    %1 = memref.alloca() : memref<100x64xi32>
    cosim.compare %0, %1 : memref<100x64xi32>
    return
}
