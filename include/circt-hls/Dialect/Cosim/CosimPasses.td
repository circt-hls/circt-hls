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

def CosimLowerWrap : FunctionPass<"cosim-lower-wrap"> {
  let summary = "Lowers cosim wrap operations";
  let description = [{
    This pass lowers cosim wrap operations into a sequence of input copying,
    target calls, and cosim.compare operations.

    Copying is inserted at the point of definition of the input values.
  }];
  let constructor = "circt_hls::cosim::createCosimLowerWrapPass()";
}

#endif // CIRCT_DIALECT_COSIM_COSIMPASSES_TD
