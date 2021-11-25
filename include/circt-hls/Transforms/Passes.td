//===-- Passes.td - Transforms pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes within the Transforms/ directory.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_HLS_TRANSFORMS_PASSES
#define CIRCT_HLS_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def AsyncifyCalls : Pass<"asyncify-calls", "ModuleOp"> {
  let summary = "Async-ifies calls to a function (HLT)";
  let description = [{
    Async-ifies calls to an HLT kernel function by decoupling the call
    and await of the function. Asyncification works within simple SCF loops
    with no inter-loop dependencies.}];
  let constructor = "circt_hls::createAsyncifyCallsPass()";
  let dependentDialects = ["scf::SCFDialect"];
  let options = [
      Option<"functionName", "function", "std::string", "",
      /*description=*/"The name of the called function to asyncify.">
  ];
}

#endif // CIRCT_HLS_TRANSFORMS_PASSES
