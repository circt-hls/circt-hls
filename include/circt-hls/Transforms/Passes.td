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

def AffineScalRep : Pass<"affine-scalrep", "mlir::func::FuncOp"> {
  let summary = "Additional scalar replacement transforms";
  let description = [{
    Additional scalar replacement transforms on top of upstream MLIR. It
    forwards stores to returns, which can optimize away memrefs allocated
    for returned tensors during bufferization.
  }];
  let constructor = "circt_hls::createAffineScalRepPass()";
}


#endif // CIRCT_HLS_TRANSFORMS_PASSES
