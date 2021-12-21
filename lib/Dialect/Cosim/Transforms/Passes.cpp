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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt_hls;
using namespace cosim;

static bool isMutable(Type t) {
  return llvm::TypeSwitch<Type, bool>(t)
      .Case<MemRefType>([&](auto) { return true; })
      .Default([&](auto) { return false; });
}

// Some trickery to determine the relative difference between two operations in
// a block (std::distance with just the two ops may infinitely loop).
static int opPosDiff(Operation *op1, Operation *op2) {
  assert(op1->getBlock() == op2->getBlock());
  auto *block = op1->getBlock();
  auto &entryOp = block->front();

  return static_cast<int>(
             std::distance(entryOp.getIterator(), op2->getIterator())) -
         static_cast<int>(
             std::distance(entryOp.getIterator(), op1->getIterator()));
}

static Value copyAtLastMutationBefore(Value v, Operation *beforeOp,
                                      PatternRewriter &rewriter) {
  auto ip = rewriter.saveInsertionPoint();

  MemRefType memrefType = v.getType().dyn_cast<MemRefType>();
  assert(memrefType &&
         "Does anything else but mutations on memrefs make sense?");

  Operation *lastMutation = nullptr;
  for (auto use : v.getUsers()) {
    // If use is inside another region, recurse upwards until we've found a
    // shared region between the use and beforeOp
    Operation *useRegionOp = use;
    while (useRegionOp->getParentRegion() != beforeOp->getParentRegion())
      useRegionOp = useRegionOp->getParentOp();

    if (useRegionOp == beforeOp)
      continue;
    if (opPosDiff(useRegionOp, beforeOp) < 0)
      continue; // 'use' comes after 'beforeOp'
    if (!lastMutation)
      lastMutation = useRegionOp;
    else if (opPosDiff(lastMutation, useRegionOp) > 0)
      lastMutation = useRegionOp;
  }
  if (!lastMutation) {
    // No users found, just copy at the creation point of 'v'
    rewriter.setInsertionPointAfterValue(v);
  } else
    rewriter.setInsertionPointAfter(lastMutation);

  // Copy the value (again, memref only supported for now)
  auto memrefCopy = rewriter.create<memref::AllocOp>(v.getLoc(), memrefType);
  rewriter.create<memref::CopyOp>(v.getLoc(), v, memrefCopy.getResult());

  rewriter.restoreInsertionPoint(ip);
  return memrefCopy;
}

static void compareToRefAfterOp(Value ref, Value cosim, Operation *op,
                                PatternRewriter &rewriter) {
  auto ip = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointAfter(op);
  rewriter.create<cosim::CompareOp>(ref.getLoc(), ref, cosim);
  rewriter.restoreInsertionPoint(ip);
}

namespace {

struct ConvertCallPattern : OpRewritePattern<cosim::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cosim::CallOp op,
                                PatternRewriter &rewriter) const override {
    // Create definitions for targets if they don't exist
    auto module = op->getParentOfType<ModuleOp>();

    // First, grab the function operations for the targets.
    std::map<std::string, mlir::FuncOp> targetFunctions;
    for (auto target : op.targets()) {
      auto targetName = target.cast<StringAttr>().strref();
      mlir::FuncOp targetFunc = module.lookupSymbol<mlir::FuncOp>(targetName);
      assert(targetFunc &&
             "expected all functions to be declared in the module");
      targetFunctions[targetName.str()] = targetFunc;
    }
    auto refName = op.ref().cast<StringAttr>().str();
    mlir::FuncOp refFunc = module.lookupSymbol<mlir::FuncOp>(refName);
    assert(refFunc && "expected all functions to be declared in the module");
    targetFunctions[refName] = refFunc;

    std::map<std::string, SmallVector<Value>> targetOperands;

    // Create copies for any mutable inputs to the target functions
    for (auto target : op.targets()) {
      auto targetStr = target.cast<StringAttr>().strref().str();
      for (auto operand : op.getOperands()) {
        if (isMutable(operand.getType()))
          targetOperands[targetStr].push_back(
              copyAtLastMutationBefore(operand, op, rewriter));
        else
          targetOperands[targetStr].push_back(operand);
      }
    }

    // Add the reference function to targetOperands and just use the original
    // inputs
    for (auto operand : op.getOperands())
      targetOperands[op.func().str()].push_back(operand);

    // Create calls to the targets
    std::map<std::string, mlir::CallOp> targetCalls;
    for (auto target : targetOperands) {
      targetCalls[target.first] = rewriter.create<mlir::CallOp>(
          op.getLoc(), targetFunctions.at(target.first), target.second);
    }

    // Emit cosim comparison between the reference function and the target
    // functions.
    mlir::CallOp refCall = targetCalls[op.ref().cast<StringAttr>().str()];
    for (auto &call : targetCalls) {
      if (call.second == refCall)
        continue;

      // Emit comparison operations on mutable inputs
      for (auto [refOperand, targetOperand] :
           llvm::zip(refCall.getOperands(), call.second.getOperands())) {
        if (isMutable(refOperand.getType()))
          compareToRefAfterOp(refOperand, targetOperand, call.second, rewriter);
      }

      // Emit comparison operations on results
      for (auto [refRes, targetRes] :
           llvm::zip(refCall.getResults(), call.second.getResults()))
        compareToRefAfterOp(refRes, targetRes, call.second, rewriter);
    }

    // Erase the cosim.call operation
    rewriter.replaceOp(op, refCall.getResults());

    return success();
  }
};

struct CosimLowerWrapPass : public CosimLowerWrapBase<CosimLowerWrapPass> {
  void runOnFunction() override {

    for (auto cosimCallOp : getOperation().getOps<cosim::CallOp>())
      createExternalSymbols(cosimCallOp);

    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<ConvertCallPattern>(ctx);
    ConversionTarget target(*ctx);
    target.addIllegalOp<cosim::CallOp>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<mlir::BuiltinDialect>();
    target.addLegalDialect<mlir::StandardOpsDialect>();
    target.addLegalDialect<arith::ArithmeticDialect>();
    target.addLegalDialect<cosim::CosimDialect>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  };

  void createExternalSymbols(cosim::CallOp op) {
    auto module = op->getParentOfType<ModuleOp>();
    auto funcOp = op->getParentOfType<mlir::FuncOp>();
    FunctionType opFuncType = FunctionType::get(
        op.getContext(), op.getOperandTypes(), op.getResultTypes());
    ImplicitLocOpBuilder builder(module.getLoc(), op.getContext());
    builder.setInsertionPoint(funcOp);
    for (auto target : op.targets()) {
      auto targetName = target.cast<StringAttr>().strref();
      mlir::FuncOp targetFunc = module.lookupSymbol<mlir::FuncOp>(targetName);
      if (!targetFunc) {
        // Function doesn't exist in module; create private definition.
        builder.setInsertionPoint(funcOp);
        targetFunc =
            builder.create<mlir::FuncOp>(op.getLoc(), targetName, opFuncType);
        targetFunc.setPrivate();
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> circt_hls::cosim::createCosimLowerWrapPass() {
  return std::make_unique<CosimLowerWrapPass>();
}
