// RUN: mlir-resolve --file1 %s --file2 %s.def

// CHECK: module  {
// CHECK:   func @foo(%arg0: i32) {
// CHECK:     %0 = call @bar(%arg0) : (i32) -> i32
// CHECK:     return
// CHECK:   }
// CHECK:   func @bar(%arg0: i32) -> i32 {
// CHECK:     %c0_i32 = arith.constant 0 : i32
// CHECK:     %0 = arith.addi %arg0, %c0_i32 : i32
// CHECK:     return %0 : i32
// CHECK:   }
// CHECK: }
module {
  func @foo(%arg0: i32) {
    %0 = call @bar(%arg0) : (i32) -> (i32)
    return
  }
  func private @bar(i32) -> (i32)
}