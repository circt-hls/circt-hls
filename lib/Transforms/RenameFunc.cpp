//===- MaxSSA.cpp - Maximal SSA form conversion ------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the maximal SSA form conversion pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/TypeSwitch.h"

#include <set>

using namespace mlir;
using namespace circt_hls;

namespace {

struct RenameFunctionPass : public RenameFunctionBase<RenameFunctionPass> {
public:
  void runOnOperation() override {
    auto moduleOp = getOperation();

    auto targetFuncOp = moduleOp.lookupSymbol<func::FuncOp>(targetFunc);
    if (!targetFuncOp) {
      emitError(moduleOp.getLoc())
          << "Target function " << targetFunc << " not found in module";
      return signalPassFailure();
    }

    SymbolTable::setSymbolName(targetFuncOp, renameTo);
  }
};

} // namespace

namespace circt_hls {
std::unique_ptr<mlir::Pass> createRenameFunctionPass() {
  return std::make_unique<RenameFunctionPass>();
}
} // namespace circt_hls
