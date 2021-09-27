module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main(%arg0: i32) -> i32 {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c1_i32 = constant 1 : i32
    %c0_i32 = constant 0 : i32
    %0 = memref.alloca() : memref<8xi32>
    %1 = index_cast %arg0 : i32 to index
    %2:2 = scf.while (%arg1 = %c0, %arg2 = %c0_i32) : (index, i32) -> (index, i32) {
      %4 = cmpi slt, %arg1, %1 : index
      scf.condition(%4) %arg1, %arg2 : index, i32
    } do {
    ^bb0(%arg1: index, %arg2: i32):  // no predecessors
      %4 = addi %arg1, %c1 : index
      %5 = index_cast %arg2 : i32 to index
      memref.store %arg2, %0[%5] : memref<8xi32>
      %6 = addi %arg2, %c1_i32 : i32
      scf.yield %4, %6 : index, i32
    }
    %3:3 = scf.while (%arg1 = %c0, %arg2 = %c0_i32, %arg3 = %c0_i32) : (index, i32, i32) -> (index, i32, i32) {
      %4 = cmpi slt, %arg1, %1 : index
      scf.condition(%4) %arg1, %arg2, %arg3 : index, i32, i32
    } do {
    ^bb0(%arg1: index, %arg2: i32, %arg3: i32):  // no predecessors
      %4 = addi %arg1, %c1 : index
      %5 = memref.load %0[%arg1] : memref<8xi32>
      %6 = addi %arg2, %5 : i32
      scf.yield %4, %6, %6 : index, i32, i32
    }
    return %3#2 : i32
  }
}


