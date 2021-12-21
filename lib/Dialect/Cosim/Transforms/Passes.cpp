//===- Buffers.cpp - buffer materialization passes --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt-hls/Dialect/Cosim/CosimOps.h"
#include "circt-hls/Dialect/Cosim/CosimPasses.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt_hls;
using namespace cosim;

namespace {

struct CosimLowerWrapPass : public CosimLowerWrapBase<CosimLowerWrapPass> {
  void runOnFunction() override{

  };
};

} // namespace

std::unique_ptr<mlir::Pass> circt_hls::cosim::createCosimLowerWrapPass() {
  return std::make_unique<CosimLowerWrapPass>();
}
