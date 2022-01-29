
// In this example, the fifos must buffer values since the other path to the
// adder has sequential buffers.
// The balance should also be big enough to make the fifo buffers full.
module {
  handshake.func @fifo_buffers(%arg0: i32, %arg1 : i32, %ctrl : none) -> (i32, none) {
    %0 = buffer [5] %arg0 {sequential = false} : i32
    %1 = buffer [50] %arg1 {sequential = true} : i32
    %2 = arith.addi %0, %1 : i32
    return %2, %ctrl : i32, none
  }
}
