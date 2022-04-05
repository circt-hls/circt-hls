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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "asyncify-calls"

using namespace mlir;
using namespace func;
using namespace circt_hls;

namespace {

// Adds value mappings to map between the iv, step, ub and lb of two loops.
static void addLoopToLoopMapping(BlockAndValueMapping &mapping, scf::ForOp from,
                                 scf::ForOp to) {
  mapping.map(from.getInductionVar(), to.getInductionVar());
  mapping.map(from.getStep(), to.getStep());
  mapping.map(from.getLowerBound(), to.getLowerBound());
  mapping.map(from.getUpperBound(), to.getUpperBound());
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
                                 BlockAndValueMapping &mapping, Operation *op,
                                 bool cloneOp = true) {
  LLVM_DEBUG(llvm::dbgs() << "Upstream cloning: " << *op << "\n");
  for (auto operand : op->getOperands()) {
    if (mapping.contains(operand))
      continue;
    auto producerOp = operand.getDefiningOp();
    if (producerOp->getParentRegion() != op->getParentRegion())
      continue;

    // Recursively ensure that all operands of the producerOp are available in
    // the mapping (post-order traversal of the DFG).
    recurseCloneUpstream(rewriter, mapping, producerOp);
  }

  if (cloneOp) {
    // All operands are now either available in the parent region or in the
    // mapping. Clone the current op
    auto clonedOp = rewriter.clone(*op, mapping);

    // Extend mapping with the results of the cloned op.
    mapAllResults(mapping, op, clonedOp);
  }
  LLVM_DEBUG(llvm::dbgs() << "Finished upstream cloning: " << *op << "\n");
}

// Like recurseCloneUpstream, but clones based on the dataflow graph following
// the result of operations.
static void recurseCloneDownstream(ConversionPatternRewriter &rewriter,
                                   BlockAndValueMapping &mapping, Operation *op,
                                   bool cloneOp = true,
                                   bool cloneUpstream = true) {
  LLVM_DEBUG(llvm::dbgs() << "Downstream cloning: " << *op << "\n");

  if (cloneUpstream) {
    // Recursively ensure that all operands of the userOp are available in
    // the mapping (post-order traversal of the DFG). Do not clone op since it
    // will be cloned from here.
    recurseCloneUpstream(rewriter, mapping, op, false);
  }

  if (cloneOp) {
    // Clone operation into insertion point
    auto clonedOp = rewriter.clone(*op, mapping);

    // Extend mapping with the results of the cloned user.
    mapAllResults(mapping, op, clonedOp);
  }

  // Recursively ensure that all downstream users of this op are cloned as well.
  for (auto operand : op->getResults()) {
    for (auto userOp : operand.getUsers()) {
      recurseCloneDownstream(rewriter, mapping, userOp);
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "Finished downstream cloning: " << *op << "\n");
}

// Recursively erases all downstream users of the results of op, and finally the
// op itself.
static void recurseCleanDownstream(ConversionPatternRewriter &rewriter,
                                   Operation *op, bool eraseOp = true) {
  for (auto res : op->getResults()) {
    for (auto user : res.getUsers()) {
      recurseCleanDownstream(rewriter, user);
    }
  }
  if (eraseOp)
    rewriter.eraseOp(op);
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

    auto callOps = op.getOps<mlir::func::CallOp>();
    if (!llvm::all_of(callOps, [&](auto callOp) {
          return callOp.getCallee() == targetName;
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

    rewriter.startRootUpdate(op);

    // Create the call and await ops within the source for loop, simplifying
    // result remapping.
    rewriter.setInsertionPoint(sourceCallOp);
    rewriter.create<CallOp>(op.getLoc(), callFunc, sourceCallOp.operands());
    auto awaitOp = rewriter.replaceOpWithNewOp<CallOp>(sourceCallOp, awaitFunc,
                                                       ValueRange());

    // Create the await loop after the call loop.
    rewriter.setInsertionPointAfter(op);
    auto awaitLoop = rewriter.create<scf::ForOp>(
        op.getLoc(), op.getLowerBound(), op.getUpperBound(), op.getStep());

    // Move result dependencies (and upstream dependencies of downstream ops) of
    // the call to the await loop.
    BlockAndValueMapping mapping;
    addLoopToLoopMapping(mapping, op, awaitLoop);
    rewriter.setInsertionPoint(awaitLoop.getBody(),
                               awaitLoop.getBody()->begin());
    auto clonedAwaitOp = rewriter.clone(*awaitOp, mapping);
    mapAllResults(mapping, sourceCallOp, clonedAwaitOp);
    // We recurse clone from the source call op since the SSA value replacements
    // have yet to be materialized.
    recurseCloneDownstream(rewriter, mapping, sourceCallOp, /*cloneOp=*/false,
                           /*cloneUpstream=*/false);

    // Cleanup by erasing everything that is downstream from the sourceCallOp in
    // the call loop. eraseOp is false due to sourceCallOp already being
    // replaced by replaceOpWithNewOp.
    recurseCleanDownstream(rewriter, sourceCallOp, /*eraseOp=*/false);
    // erase the temporary await op.
    rewriter.eraseOp(awaitOp);
    rewriter.finalizeRootUpdate(op);
    return success();
  }
};

struct CallOpConversion : public AsyncConversionPattern<mlir::func::CallOp> {
  using AsyncConversionPattern::AsyncConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op.getCallee().equals(targetName) &&
           "Legalizer should not allow this");
    assert(isa<FuncOp>(op->getParentOp()) &&
           "Call ops nested within anything but a FuncOp should have been "
           "converted prior to this conversion pattern applying. This pattern "
           "intends to only modify CallOps nested directly within a function "
           "body.");

    auto awaitCall = rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        op, awaitFunc, ValueRange());
    rewriter.setInsertionPoint(awaitCall);
    rewriter.create<mlir::func::CallOp>(op.getLoc(), callFunc,
                                        adaptor.getOperands());
    return success();
  }
};

struct AsyncifyCallsPass : public AsyncifyCallsBase<AsyncifyCallsPass> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();
    auto module = getOperation();

    if (functionName.empty()) {
      // Try to infer the callee function.
      StringRef callee;
      auto res = getOperation().walk([&](mlir::func::CallOp op) {
        if (!callee.empty() && !callee.equals(op.getCallee()))
          return WalkResult::interrupt();
        callee = op.getCallee();
        return WalkResult::advance();
      });

      if (res.wasInterrupted()) {
        getOperation().emitError()
            << "Multiple functions called within the body of a function. Must "
               "provide a --function argument to determine the callee to be "
               "asyncified.";
        return signalPassFailure();
      }
      functionName = callee.str();
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
  // Expecting affine loops to be lowered to SCF
  target.addIllegalDialect<mlir::AffineDialect>();
  target.addLegalDialect<scf::SCFDialect>();
  target.addLegalDialect<memref::MemRefDialect>();
  target.addLegalDialect<mlir::BuiltinDialect>();
  target.addLegalDialect<mlir::cf::ControlFlowDialect>();
  target.addLegalDialect<arith::ArithmeticDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addDynamicallyLegalOp<mlir::func::CallOp>([&](CallOp op) {
    // We expect the target function to be removed after asyncification.
    return op.getCallee() != functionName;
  });
  target.addDynamicallyLegalOp<scf::ForOp>([&](scf::ForOp op) {
    // Loops are legal when they don't have a call to the target function OR
    // there exists a _call/_await pair within the loop. The latter condition is
    // true when the dynamic legalizer is called during conversion, where both
    // the target function call and the decoupled function calls exist.
    auto callOps = op.getOps<CallOp>();
    auto hasAsyncPair = [&](StringRef callee) {
      bool call = llvm::any_of(callOps, [&](CallOp callOp) {
        return callOp.getCallee().equals((callee + "_call").str());
      });
      bool await = llvm::any_of(callOps, [&](CallOp callOp) {
        return callOp.getCallee().equals((callee + "_await").str());
      });
      return call && await;
    };

    for (auto callOp : callOps) {
      if (!callOp.getCallee().equals(functionName))
        continue;

      // Is there an async call?
      if (!hasAsyncPair(callOp.getCallee()))
        return false;
    }
    return true;

    return llvm::none_of(callOps, [&](CallOp callOp) {
      return callOp.getCallee().equals(functionName);
    });
  });
}

void AsyncifyCallsPass::createExternalSymbols(FuncOp sourceOp) {
  auto *ctx = &getContext();
  auto module = getOperation();

  FunctionType type = sourceOp.getFunctionType();
  ImplicitLocOpBuilder builder(module.getLoc(), ctx);
  builder.setInsertionPoint(sourceOp);
  callOp = builder.create<FuncOp>(
      (sourceOp.getName() + "_call").str(),
      FunctionType::get(ctx, type.getInputs(), TypeRange()));
  awaitOp = builder.create<FuncOp>(
      (sourceOp.getName() + "_await").str(),
      FunctionType::get(ctx, TypeRange(), type.getResults()));
  callOp.setPrivate();
  awaitOp.setPrivate();
}

} // namespace

namespace circt_hls {
std::unique_ptr<mlir::Pass> createAsyncifyCallsPass() {
  return std::make_unique<AsyncifyCallsPass>();
}
} // namespace circt_hls
