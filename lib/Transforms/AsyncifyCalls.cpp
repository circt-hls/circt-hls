//===- AsyncifyCalls.cpp - Asyncification -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the Asyncification flattening pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace circt_hls;

namespace {

// Adds value mappings to map between the iv, step, ub and lb of two loops.
static void addLoopToLoopMapping(BlockAndValueMapping &mapping, scf::ForOp from,
                                 scf::ForOp to) {
  mapping.map(from.getInductionVar(), to.getInductionVar());
  mapping.map(from.step(), to.step());
  mapping.map(from.lowerBound(), to.lowerBound());
  mapping.map(from.upperBound(), to.upperBound());
}

static void mapAllResults(BlockAndValueMapping &mapping, Operation *from,
                          Operation *to) {
  for (auto res : llvm::zip(from->getResults(), to->getResults()))
    mapping.map(std::get<0>(res), std::get<1>(res));
}

// Post-order dataflow analysis which recursively clones predecessor operations
// within the same region. The rewriter is expected to maintain the insertion
// point. Each cloned operation extends the value mapping - this is necessary to
// avoid re-cloning operations which fan-out to multiple operations.
static void recurseCloneUpstream(ConversionPatternRewriter &rewriter,
                                 BlockAndValueMapping &mapping, Operation *op) {
  for (auto operand : op->getOperands()) {
    if (mapping.contains(operand))
      continue;
    auto producerOp = operand.getDefiningOp();
    if (producerOp->getParentRegion() != op->getParentRegion())
      continue;

    // Recursively ensure that all operands of the producerOp are available in
    // the mapping (post-order traversal of the DFG).
    recurseCloneUpstream(rewriter, mapping, producerOp);

    // All operands are now either available in the parent region or in the
    // mapping.
    auto clonedProducerOp = rewriter.clone(*producerOp, mapping);

    // Extend mapping with the results of the cloned producer.
    mapAllResults(mapping, producerOp, clonedProducerOp);
  }
}

// Like recurseCloneUpstream, but clones based on the dataflow graph following
// the result of operations.
static void recurseCloneDownstream(ConversionPatternRewriter &rewriter,
                                   BlockAndValueMapping &mapping,
                                   Operation *op) {
  for (auto operand : op->getResults()) {
    for (auto userOp : operand.getUsers()) {
      // Recursively ensure that all operands of the userOp are available in
      // the mapping (post-order traversal of the DFG).
      recurseCloneUpstream(rewriter, mapping, userOp);

      // All operands are now either available in the parent region or in the
      // mapping. Clone the user
      auto clonedUserOp = rewriter.clone(*userOp, mapping);

      // Extend mapping with the results of the cloned user.
      mapAllResults(mapping, clonedUserOp, clonedUserOp);

      // Recursively ensure that downstream operations are cloned
      recurseCloneDownstream(rewriter, mapping, userOp);
    }
  }
}

template <typename TOp>
struct AsyncConversionPattern : public OpConversionPattern<TOp> {
  AsyncConversionPattern(MLIRContext *ctx, FuncOp callFunc, FuncOp awaitFunc,
                         StringRef targetName)
      : OpConversionPattern<TOp>(ctx), callFunc(callFunc), awaitFunc(awaitFunc),
        targetName(targetName) {}
  FuncOp callFunc;
  FuncOp awaitFunc;
  StringRef targetName;
};

struct ForOpConversion : public AsyncConversionPattern<scf::ForOp> {
  using AsyncConversionPattern::AsyncConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only support the trivial case of having a single call to the target
    // function within the loop body. We allow multiple instructions within the
    // loop body, but not calls to other functions.

    auto callOps = op.getOps<mlir::CallOp>();
    if (!llvm::all_of(callOps, [&](auto callOp) {
          return callOp.callee() == targetName;
        }))
      return op.emitOpError()
             << "Cannot transform a for loop which calls both the target "
                "function and non-target functions.";

    if (std::distance(callOps.begin(), callOps.end()) != 1)
      return op.emitOpError() << "Cannot transform a for loop with multiple "
                                 "calls to the target function";

    if (!op.getIterOperands().empty())
      return op.emitOpError()
             << "Cannot transform a for loop with iter arguments";

    CallOp sourceCallOp = *callOps.begin();

    // Create the call and await ops within the source for loop, simplifying
    // result remapping.
    rewriter.setInsertionPoint(sourceCallOp);
    auto callOp =
        rewriter.create<CallOp>(op.getLoc(), callFunc, sourceCallOp.operands());
    auto awaitOp = rewriter.replaceOpWithNewOp<CallOp>(sourceCallOp, awaitFunc,
                                                       ValueRange());

    // Create the call and await loops
    auto callLoop = rewriter.create<scf::ForOp>(op.getLoc(), op.lowerBound(),
                                                op.upperBound(), op.step());
    auto awaitLoop = rewriter.create<scf::ForOp>(op.getLoc(), op.lowerBound(),
                                                 op.upperBound(), op.step());

    // Move argument dependencies of the call to the callLoop
    BlockAndValueMapping mapping;
    addLoopToLoopMapping(mapping, op, callLoop);

    // move downstream ops (and dependencies) to the awaitLoop

    return success();
  }
};

struct CallOpConversion : public AsyncConversionPattern<mlir::CallOp> {
  using AsyncConversionPattern::AsyncConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    assert(op.callee() != targetName && "Legalizer should not allow this");
    assert(isa<FuncOp>(op->getParentOp()) &&
           "Call ops nested within anything but a FuncOp should have been "
           "converted prior to this conversion pattern applying. This pattern "
           "intends to only modify CallOps nested directly within a function "
           "body.");

    auto awaitCall =
        rewriter.replaceOpWithNewOp<mlir::CallOp>(op, awaitFunc, ValueRange());
    rewriter.setInsertionPoint(awaitCall);
    rewriter.create<mlir::CallOp>(op.getLoc(), callFunc, adaptor.getOperands());
    return success();
  }
};

struct AsyncifyCallsPass : public AsyncifyCallsBase<AsyncifyCallsPass> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();
    auto module = getOperation();

    if (functionName.empty()) {
      getOperation().emitError() << "Must provide a --function argument";
      return signalPassFailure();
    }

    FuncOp source = module.lookupSymbol<FuncOp>(functionName);
    if (!source) {
      getOperation().emitError()
          << "function '" << functionName << "' not found in module";
      return signalPassFailure();
    }

    // Create external symbols for the asyncify calls.
    createExternalSymbols(source);

    RewritePatternSet patterns(ctx);
    patterns.insert<ForOpConversion, CallOpConversion>(ctx, callOp, awaitOp,
                                                       functionName);

    ConversionTarget target(*ctx);
    addTargetLegalizations(target);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  };

  void createExternalSymbols(FuncOp sourceOp);
  void addTargetLegalizations(ConversionTarget &target);

private:
  FuncOp callOp;
  FuncOp awaitOp;
};

void AsyncifyCallsPass::addTargetLegalizations(ConversionTarget &target) {
  target.addLegalDialect<scf::SCFDialect>();
  target.addDynamicallyLegalOp<mlir::CallOp>([&](CallOp op) {
    // We expect the target function to be removed after asyncification.
    return op.callee() != functionName;
  });
  target.addDynamicallyLegalOp<scf::ForOp>([&](scf::ForOp op) {
    // Loops are legal when they don't have a call to the target function.
    auto callOps = op.getOps<CallOp>();
    return llvm::none_of(callOps, [&](CallOp callOp) {
      return callOp.callee() != functionName;
    });
  });
}

void AsyncifyCallsPass::createExternalSymbols(FuncOp sourceOp) {
  auto *ctx = &getContext();
  auto module = getOperation();

  FunctionType type = sourceOp.getType();
  ImplicitLocOpBuilder builder(module.getLoc(), ctx);
  callOp = builder.create<FuncOp>(
      (sourceOp.getName() + "_call").str(),
      FunctionType::get(ctx, type.getInputs(), TypeRange()));
  awaitOp = builder.create<FuncOp>(
      (sourceOp.getName() + "_await").str(),
      FunctionType::get(ctx, TypeRange(), type.getResults()));
}

} // namespace

namespace circt_hls {
std::unique_ptr<mlir::Pass> createAsyncifyCallsPass() {
  return std::make_unique<AsyncifyCallsPass>();
}
} // namespace circt_hls
