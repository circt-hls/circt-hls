// RUN: mlir-clang --S --function=* --memref-fullrank %s | FileCheck %s

// CHECK-LABEL:   func @foo(%arg0: memref<20xi32>) -> i32
// CHECK:     %0 = affine.load %arg0[0] : memref<20xi32>
// CHECK:     return %0 : i32
// CHECK:   }
int foo(int x[20]) { return x[0]; }
