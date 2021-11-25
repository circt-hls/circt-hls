// RUN: hls-opt -split-input-file -asyncify-calls %s -verify-diagnostics

// expected-error @+1 {{Multiple functions called within the body of a function. Must provide a --function argument to determine the callee to be asyncified.}}
module {
  func private @foo()
  func private @bar()
  func @uninferable() {
    call @foo() : () -> ()
    call @bar() : () -> ()
    return
  }
}

