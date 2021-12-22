// RUN: hls-opt --split-input-file --cosim-lower-call %s | FileCheck %s

// CHECK-LABEL:   func @wrap_simple() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           call @foo(%[[VAL_0]], %[[VAL_1]]) : (i32, i32) -> ()
// CHECK:           call @foo_hlt(%[[VAL_0]], %[[VAL_1]]) : (i32, i32) -> ()
// CHECK:           return
// CHECK:         }
module {
  func @wrap_simple() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    cosim.call @foo(%c0_i32, %c1_i32) : (i32, i32) -> () {
      targets = ["foo_hlt"],
      ref = "foo"
    }
    return
  }
  func private @foo(i32, i32) -> ()
}

// -----

// CHECK-LABEL:   func @wrap_simple_with_ret() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = call @foo(%[[VAL_0]], %[[VAL_1]]) : (i32, i32) -> i32
// CHECK:           %[[VAL_3:.*]] = call @foo_hlt(%[[VAL_0]], %[[VAL_1]]) : (i32, i32) -> i32
// CHECK:           cosim.compare %[[VAL_2]], %[[VAL_3]] : i32
// CHECK:           return
// CHECK:         }
module {
  func @wrap_simple_with_ret() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    cosim.call @foo(%c0_i32, %c1_i32) : (i32, i32) -> (i32)
    {
      targets = ["foo_hlt"],
      ref = "foo"
    }
    return
  }
  func private @foo(i32, i32) -> (i32)
}

// -----

// CHECK-LABEL:   func @wrap_simple_memref() {
// CHECK:           %[[VAL_0:.*]] = memref.alloca() : memref<100xi32>
// CHECK:           %[[VAL_1:.*]] = memref.alloc() : memref<100xi32>
// CHECK:           memref.copy %[[VAL_0]], %[[VAL_1]] : memref<100xi32> to memref<100xi32>
// CHECK:           call @foo(%[[VAL_0]]) : (memref<100xi32>) -> ()
// CHECK:           call @foo_hlt(%[[VAL_1]]) : (memref<100xi32>) -> ()
// CHECK:           cosim.compare %[[VAL_0]], %[[VAL_1]] : memref<100xi32>
// CHECK:           return
// CHECK:         }
module {
  func @wrap_simple_memref() {
    %0 = memref.alloca() : memref<100xi32>
    cosim.call @foo(%0) : (memref<100xi32>) -> ()
    {
      targets = ["foo_hlt"],
      ref = "foo"
    }
    return
  }
  func private @foo(memref<100xi32>) -> ()
}

// -----

// CHECK-LABEL:   func @wrap_initialized_memref() {
// CHECK:           %[[VAL_0:.*]] = memref.alloca() : memref<100xi32>
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 100 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           scf.for %[[VAL_4:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_3]] {
// CHECK:             %[[VAL_5:.*]] = arith.index_cast %[[VAL_4]] : index to i32
// CHECK:             memref.store %[[VAL_5]], %[[VAL_0]]{{\[}}%[[VAL_4]]] : memref<100xi32>
// CHECK:           }
// CHECK:           %[[VAL_6:.*]] = memref.alloc() : memref<100xi32>
// CHECK:           memref.copy %[[VAL_0]], %[[VAL_6]] : memref<100xi32> to memref<100xi32>
// CHECK:           %[[VAL_7:.*]] = call @foo(%[[VAL_0]]) : (memref<100xi32>) -> i32
// CHECK:           %[[VAL_8:.*]] = call @foo_std(%[[VAL_6]]) : (memref<100xi32>) -> i32
// CHECK:           cosim.compare %[[VAL_7]], %[[VAL_8]] : i32
// CHECK:           cosim.compare %[[VAL_0]], %[[VAL_6]] : memref<100xi32>
// CHECK:           return
// CHECK:         }
module {
  func @wrap_initialized_memref() {
    %0 = memref.alloca() : memref<100xi32>
    %c0 = arith.constant 0 : index
    %c100 = arith.constant 100 : index
    %c1 = arith.constant 1 : index
    scf.for %arg0 = %c0 to %c100 step %c1 {
      %1 = arith.index_cast %arg0 : index to i32
      memref.store %1, %0[%arg0] : memref<100xi32>
    }
    %1 = cosim.call @foo(%0) : (memref<100xi32>) -> (i32) 
      {ref = "foo", targets = ["foo_std"]}
    return
  }
  func private @foo(memref<100xi32>) -> i32
}

// -----

// CHECK-LABEL:   func @wrap_multiple_targets() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = call @foo(%[[VAL_0]], %[[VAL_1]]) : (i32, i32) -> i32
// CHECK:           %[[VAL_3:.*]] = call @foo_1(%[[VAL_0]], %[[VAL_1]]) : (i32, i32) -> i32
// CHECK:           cosim.compare %[[VAL_2]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_4:.*]] = call @foo_2(%[[VAL_0]], %[[VAL_1]]) : (i32, i32) -> i32
// CHECK:           cosim.compare %[[VAL_2]], %[[VAL_4]] : i32
// CHECK:           return
// CHECK:         }
// CHECK:         func private @foo(i32, i32) -> i32
module {
  func @wrap_multiple_targets() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = cosim.call @foo(%c0_i32, %c1_i32) : (i32, i32) -> (i32)
    {
      targets = ["foo_1", "foo_2"],
      ref = "foo"
    }
    return
  }
  func private @foo(i32, i32) -> (i32)
}


// -----

// This tests a bug where the ref call was emitted after the targets (and compares)
// to the ref call results.

// CHECK-LABEL:   func @ordering() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = call @foo(%[[VAL_0]], %[[VAL_1]]) : (i32, i32) -> i32
// CHECK:           %[[VAL_3:.*]] = call @a(%[[VAL_0]], %[[VAL_1]]) : (i32, i32) -> i32
// CHECK:           cosim.compare %[[VAL_2]], %[[VAL_3]] : i32 {ref_src = @foo, target_src = @a}
// CHECK:           %[[VAL_4:.*]] = call @b(%[[VAL_0]], %[[VAL_1]]) : (i32, i32) -> i32
// CHECK:           cosim.compare %[[VAL_2]], %[[VAL_4]] : i32 {ref_src = @foo, target_src = @b}
// CHECK:           return
// CHECK:         }
module {
  func @ordering() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = cosim.call @foo(%c0_i32, %c1_i32) : (i32, i32) -> i32 {ref = "foo", targets = ["a", "b"]}
    return
  }
  func private @foo(i32, i32) -> i32
}
