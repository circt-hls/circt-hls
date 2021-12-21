//===-- CosimPasses.td - Cosim pass definition file *- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes that work on the Cosim dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_COSIM_COSIMPASSES_TD
#define CIRCT_DIALECT_COSIM_COSIMPASSES_TD

include "mlir/Pass/PassBase.td"

def CosimLowerCall : FunctionPass<"cosim-lower-call"> {
  let summary = "Lowers cosim.call operations";
  let description = [{
    This pass lowers cosim call operations into a sequence of input copying,
    target calls, and cosim.compare operations.

    Copying is inserted at the last point of modification for mutable input values.
  }];
  let constructor = "circt_hls::cosim::createCosimLowerCallPass()";
}

def CosimLowerCompare : FunctionPass<"cosim-lower-compare"> {
  let summary = "Lowers cosim.compare operations";
  let description = [{
    This pass lowers cosim.compare operations into operations that can be further
    lowered to LLVM, and executed.

    @todo: should this be a runtime library?
  }];
  let constructor = "circt_hls::cosim::createCosimLowerComparePass()";
  let dependentDialects = [
    "mlir::scf::SCFDialect", "mlir::LLVM::LLVMDialect"
  ];
}


#endif // CIRCT_DIALECT_COSIM_COSIMPASSES_TD
