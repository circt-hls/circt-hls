// RUN: hls-opt --split-input-file --cosim-lower-wrap %s | FileCheck %s

module {
  func @wrap_simple() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    cosim.wrap {
      call @foo(%c0_i32, %c1_i32) : (i32, i32) -> ()
    } {
      targets = ["foo_hlt", "foo"],
      ref = "foo"
    }
    return
  }
  func private @foo(i32, i32) -> ()
}

// -----

module {
  func @wrap_simple_with_ret() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    cosim.wrap {
      %0 = call @foo(%c0_i32, %c1_i32) : (i32, i32) -> (i32)
    } {
      targets = ["foo_hlt", "foo"],
      ref = "foo"
    }
    return
  }
  func private @foo(i32, i32) -> (i32)
}

// -----

module {
  func @wrap_simple_memref() {
    %0 = memref.alloca() : memref<100xi32>
    cosim.wrap {
      call @foo(%0) : (memref<100xi32>) -> ()
    } {
      targets = ["foo_hlt", "foo"],
      ref = "foo"
    }
    return
  }
  func private @foo(memref<100xi32>) -> ()
}