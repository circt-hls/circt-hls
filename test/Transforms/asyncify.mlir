// RUN: hls-opt -split-input-file -asyncify-calls %s | FileCheck %s


func private @bar()

// CHECK-LABEL:   func private @bar_call()
// CHECK:         func private @bar_await()
// CHECK:         func private @bar()

// CHECK-LABEL:   func @infer_callee() {
// CHECK:           call @bar_call() : () -> ()
// CHECK:           call @bar_await() : () -> ()
// CHECK:           return
// CHECK:         }
func @infer_callee() {
  call @bar() : () -> ()
  return
}

// -----

func private @bar(i32) -> i32

// CHECK-LABEL:   func @multiple_calls(
// CHECK-SAME:                         %[[VAL_0:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_1:.*]] = arith.addi %[[VAL_0]], %[[VAL_0]] : i32
// CHECK:           call @bar_call(%[[VAL_1]]) : (i32) -> ()
// CHECK:           %[[VAL_2:.*]] = call @bar_await() : () -> i32
// CHECK:           %[[VAL_3:.*]] = arith.addi %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:           call @bar_call(%[[VAL_3]]) : (i32) -> ()
// CHECK:           %[[VAL_4:.*]] = call @bar_await() : () -> i32
// CHECK:           return %[[VAL_4]] : i32
// CHECK:         }
func @multiple_calls(%0 : i32) -> i32 {
  %1 = arith.addi %0, %0 : i32
  %2 = call @bar(%1) : (i32) -> (i32)
  %3 = arith.addi %1, %2 : i32
  %4 = call @bar(%3) : (i32) -> (i32)
  return %4 : i32
}

// -----

func private @bar(i32) -> ()

// CHECK-LABEL:   func @simple_loop(
// CHECK-SAME:                      %[[VAL_0:.*]]: i32) {
// CHECK:           %[[VAL_1:.*]] = memref.alloc() : memref<4xi32>
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           scf.for %[[VAL_5:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_4]] {
// CHECK:             %[[VAL_6:.*]] = arith.index_cast %[[VAL_5]] : index to i32
// CHECK:             call @bar_call(%[[VAL_6]]) : (i32) -> ()
// CHECK:           }
// CHECK:           scf.for %[[VAL_7:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_4]] {
// CHECK:             call @bar_await() : () -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }
func @simple_loop(%0 : i32) {
  %mem = memref.alloc() : memref<4xi32>
  %lb = arith.constant 0 : index
  %ub = arith.constant 10 : index
  %step = arith.constant 1 : index
  scf.for %i = %lb to %ub step %step {
    %i_i32 = arith.index_cast %i : index to i32
    call @bar(%i_i32) : (i32) -> ()
  }
  return
}

// -----

func private @bar(i32) -> (i32)

// CHECK-LABEL:   func @simple_loop_up_and_downstream_dep(
// CHECK-SAME:                                            %[[VAL_0:.*]]: i32) {
// CHECK:           %[[VAL_1:.*]] = memref.alloc() : memref<4xi32>
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           scf.for %[[VAL_5:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_4]] {
// CHECK:             %[[VAL_6:.*]] = arith.index_cast %[[VAL_5]] : index to i32
// CHECK:             call @bar_call(%[[VAL_6]]) : (i32) -> ()
// CHECK:           }
// CHECK:           scf.for %[[VAL_7:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_4]] {
// CHECK:             %[[VAL_8:.*]] = call @bar_await() : () -> i32
// CHECK:             %[[VAL_9:.*]] = arith.index_cast %[[VAL_7]] : index to i32
// CHECK:             %[[VAL_10:.*]] = arith.addi %[[VAL_8]], %[[VAL_9]] : i32
// CHECK:             memref.store %[[VAL_10]], %[[VAL_1]]{{\[}}%[[VAL_7]]] : memref<4xi32>
// CHECK:           }
// CHECK:           return
// CHECK:         }
func @simple_loop_up_and_downstream_dep(%0 : i32) {
  %mem = memref.alloc() : memref<4xi32>
  %lb = arith.constant 0 : index
  %ub = arith.constant 10 : index
  %step = arith.constant 1 : index
  scf.for %i = %lb to %ub step %step {
    %i_i32 = arith.index_cast %i : index to i32
    %res = call @bar(%i_i32) : (i32) -> (i32)
    %sum = arith.addi %res, %i_i32 : i32
    memref.store %sum, %mem[%i] : memref<4xi32>
  }
  return
}
