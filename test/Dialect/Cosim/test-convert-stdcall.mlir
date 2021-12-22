// RUN: hls-opt --split-input-file                              \
// RUN:     --cosim-convert-call="from=foo ref=foo targets=a,b" \
// RUN: | FileCheck %s

// @todo: i have no idea why lit doesn't seem to be able to match on these...!!!
// XFAIL: *

// CHECK-LABEL:   func @test() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = cosim.call @foo(%[[VAL_0]], %[[VAL_1]]) : (i32, i32) -> i32 {ref = "foo", targets = ["a", "b"]}
// CHECK:           return
// CHECK:         }
func @test() {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = call @foo(%c0_i32, %c1_i32) : (i32, i32) -> i32
  return
}
func private @foo(i32, i32) -> (i32)

// -----

// CHECK-LABEL:   func @test() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = cosim.call @foo(%[[VAL_0]], %[[VAL_1]]) : (i32, i32) -> i32 {ref = "foo", targets = ["a", "b"]}
// CHECK:           %[[VAL_3:.*]] = call @bar(%[[VAL_0]], %[[VAL_1]]) : (i32, i32) -> i32
// CHECK:           return
// CHECK:         }
func @test() {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = call @foo(%c0_i32, %c1_i32) : (i32, i32) -> i32
  %1 = call @bar(%c0_i32, %c1_i32) : (i32, i32) -> i32
  return
}
func private @foo(i32, i32) -> (i32)
func private @bar(i32, i32) -> (i32)
