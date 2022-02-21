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

def AffineScalRep : Pass<"affine-scalrep", "mlir::FuncOp"> {
  let summary = "Additional scalar replacement transforms";
  let description = [{
    Additional scalar replacement transforms on top of upstream MLIR. It
    forwards stores to returns, which can optimize away memrefs allocated
    for returned tensors during bufferization.
  }];
  let constructor = "circt_hls::createAffineScalRepPass()";
}

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

def MaxSSAForm : Pass<"max-ssa", "mlir::FuncOp"> {
  let summary = "Converts a program into maximal SSA form";
  let description = [{
    This pass converts a program into maximal SSA form. In this form, any value
    referenced within a block is also defined within the block. This is done by
    adding block arguments to all basic block dominance chains which may lead
    to an operation that relied on referencing a Value based on basic block
    dominance.
    
    This pass is useful in dataflow-style programming models since it renders
    all data flow within the program explicit (through block arguments) instead
    of implicit (through block dominance).

    This pass only works on Standard-level IR, in that it expects all operations
    (and blocks) within a FuncOp to be within the same region. Furthermore, it
    is assumed that any value referenced by any operation is eligible to be
    passed around as a block argument.
  }];
  let constructor = "circt_hls::createMaxSSAFormPass()";
  let options = [
    ListOption<"ignoredDialects", "ignore-dialects", "std::string",
      "List of ignored dialects. If a values' type is defined by an ignored "
      "dialect, the value will be ignored during SSA maximization.",
      "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">,
    Option<"ignoreMemref", "ignore-memref", "bool", "false",
      "Ignore memref values in SSA maximization.">
  ];
}

def PushConstants : Pass<"push-constants", "mlir::FuncOp"> {
  let summary = "Push constants into the basic blocks where they are referenced";
  let description = [{
    This pass moves constant definitions into the basic blocks where they are referenced.
    If a constant is referenced in multiple basic blocks, a copy of the constant
    is inserted in each basic block.
  }];
  let constructor = "circt_hls::createPushConstantsPass()";
}

def RenameFunction : Pass<"rename-func", "ModuleOp"> {
  let summary = "Change a builtin.func name";
  let description = [{
    Change a builtin.func name.
  }];
  let constructor = "circt_hls::createRenameFunctionPass()";
  let options = [
    Option<"targetFunc", "f", "std::string", "",
      /*description=*/"The name of the called function to rename.">,
    Option<"renameTo", "to", "std::string", "",
      /*description=*/"The new name of the function.">
  ];

}

#endif // CIRCT_HLS_TRANSFORMS_PASSES
