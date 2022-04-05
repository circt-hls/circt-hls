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

def CosimLowerCall : Pass<"cosim-lower-call", "mlir::func::FuncOp"> {
  let summary = "Lowers cosim.call operations";
  let description = [{
    This pass lowers cosim call operations into a sequence of input copying,
    target calls, and cosim.compare operations.

    Copying is inserted at the last point of modification for mutable input values.
  }];
  let constructor = "circt_hls::cosim::createCosimLowerCallPass()";
}

def CosimLowerCompare : Pass<"cosim-lower-compare", "mlir::func::FuncOp"> {
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

def CosimConvertCall : Pass<"cosim-convert-call", "mlir::func::FuncOp"> {
  let summary = "Converts a std.call operation to a cosim.call operation";
  let description = [{
    This pass converts a std.call operation to a cosim.call operation.
  }];
  let constructor = "circt_hls::cosim::createCosimConvertCallPass()";
  let options = [
    Option<"from", "from", "std::string", "",
      /*description=*/"Convert std.call operation(s) which call 'from'.">,
    Option<"ref", "ref", "std::string", "",
      /*description=*/"The name of the reference function to use.">,
    ListOption<"targets", "targets", "std::string",
      /*description=*/"The names of the target functions to use.">
  ];
  let dependentDialects = [
    "circt_hls::cosim::CosimDialect"
  ];
}


#endif // CIRCT_DIALECT_COSIM_COSIMPASSES_TD
