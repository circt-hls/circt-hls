// RUN: hls-opt -split-input-file -asyncify-calls %s -verify-diagnostics

// expected-error @+1 {{Multiple functions func.called within the body of a function. Must provide a --func.function argument to determine the func.callee to be asyncified.}}
module {
  func.func private @foo()
  func.func private @bar()
  func.func @uninferable() {
    func.call @foo() : () -> ()
    func.call @bar() : () -> ()
    return
  }
}

