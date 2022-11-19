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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
                                PatternRewriter &rewriter,
                                Operation *refSrcOp = nullptr,
                                Operation *cosimSrcOp = nullptr) {
  FlatSymbolRefAttr refSrc, cosimSrc;

  if (!refSrcOp)
    if (auto refSrcOp = dyn_cast<mlir::func::CallOp>(ref.getDefiningOp()))
      refSrc = FlatSymbolRefAttr::get(op->getContext(), refSrcOp.getCallee());

  if (!cosimSrcOp)
    if (auto cosimSrcOp = dyn_cast<mlir::func::CallOp>(cosim.getDefiningOp()))
      cosimSrc =
          FlatSymbolRefAttr::get(op->getContext(), cosimSrcOp.getCallee());

  auto ip = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointAfter(op);
  rewriter.create<cosim::CompareOp>(ref.getLoc(), ref, cosim, refSrc, cosimSrc);
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
    std::map<std::string, mlir::func::FuncOp> targetFunctions;
    for (auto target : op.getTargets()) {
      auto targetName = target.cast<StringAttr>().strref();
      mlir::func::FuncOp targetFunc =
          module.lookupSymbol<mlir::func::FuncOp>(targetName);
      assert(targetFunc &&
             "expected all functions to be declared in the module");
      targetFunctions[targetName.str()] = targetFunc;
    }
    auto refName = op.getRef().str();
    mlir::func::FuncOp refFunc =
        module.lookupSymbol<mlir::func::FuncOp>(refName);
    assert(refFunc && "expected all functions to be declared in the module");
    targetFunctions[refName] = refFunc;

    std::map<std::string, SmallVector<Value>> targetOperands;

    // Create copies for any mutable inputs to the target functions
    for (auto target : op.getTargets()) {
      auto targetStr = target.cast<StringAttr>().strref().str();
      for (auto operand : op.getOperands()) {
        if (isMutable(operand.getType()))
          targetOperands[targetStr].push_back(
              copyAtLastMutationBefore(operand, op, rewriter));
        else
          targetOperands[targetStr].push_back(operand);
      }
    }

    std::map<std::string, mlir::func::CallOp> targetCalls;
    auto emitCall = [&](StringRef callee, ValueRange operands) {
      targetCalls[callee.str()] = rewriter.create<mlir::func::CallOp>(
          op.getLoc(), targetFunctions.at(callee.str()), operands);
    };

    // Call the reference
    emitCall(op.getRef(), op.getOperands());

    // Create calls to the targets
    for (auto target : targetOperands)
      emitCall(target.first, target.second);

    // Emit cosim comparison between the reference function and the target
    // functions.
    mlir::func::CallOp refCall = targetCalls[op.getRef().str()];
    for (auto &call : targetCalls) {
      // Ignore the reference call; all other target calls will compare against
      // this call.
      if (call.second == refCall)
        continue;

      // Emit comparison operations on mutable inputs
      for (auto [refOperand, targetOperand] :
           llvm::zip(refCall.getOperands(), call.second.getOperands())) {
        if (isMutable(refOperand.getType()))
          compareToRefAfterOp(refOperand, targetOperand, call.second, rewriter,
                              refCall, call.second);
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

struct CosimLowerCallPass : public CosimLowerCallBase<CosimLowerCallPass> {
  void runOnOperation() override {

    for (auto cosimCallOp : getOperation().getOps<cosim::CallOp>())
      createExternalSymbols(cosimCallOp);

    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<ConvertCallPattern>(ctx);
    ConversionTarget target(*ctx);
    target.addIllegalOp<cosim::CallOp>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<mlir::BuiltinDialect>();
    target.addLegalDialect<mlir::cf::ControlFlowDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<cosim::CosimDialect>();
    target.addLegalDialect<func::FuncDialect>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  };

  void createExternalSymbols(cosim::CallOp op) {
    auto module = op->getParentOfType<ModuleOp>();
    auto funcOp = op->getParentOfType<mlir::func::FuncOp>();
    FunctionType opFuncType = FunctionType::get(
        op.getContext(), op.getOperandTypes(), op.getResultTypes());
    ImplicitLocOpBuilder builder(module.getLoc(), op.getContext());
    builder.setInsertionPointAfter(funcOp);

    auto createFunc = [&](StringRef name) {
      mlir::func::FuncOp targetFunc =
          module.lookupSymbol<mlir::func::FuncOp>(name);
      if (!targetFunc) {
        // Function doesn't exist in module; create private definition.
        builder.setInsertionPoint(funcOp);
        targetFunc =
            builder.create<mlir::func::FuncOp>(op.getLoc(), name, opFuncType);
        targetFunc.setPrivate();
      }
    };

    for (auto target : op.getTargets())
      createFunc(target.cast<StringAttr>().str());
    createFunc(op.getRef());
  }
};

/// NOTE: stolen directly from the MLIR LLVM lowering examples... this stuff
/// isn't exposed any where in public libraries!
/// Return a symbol reference to the printf function, inserting it into the
/// module if necessary.
static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                           ModuleOp module) {
  auto *context = module.getContext();
  if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
    return SymbolRefAttr::get(context, "printf");

  // Create a function declaration for printf, the signature is:
  //   * `i32 (i8*, ...)`
  auto llvmI32Ty = IntegerType::get(context, 32);
  auto llvmI8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
  auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy,
                                                /*isVarArg=*/true);

  // Insert the printf function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
  return SymbolRefAttr::get(context, "printf");
}

static void insertPrintfCall(Location loc, ModuleOp module,
                             PatternRewriter &rewriter, ValueRange operands) {
  auto printf = getOrInsertPrintf(rewriter, module);
  // Resolve to the LLVMFuncOp
  auto printfFuncOp = module.lookupSymbol<LLVM::LLVMFuncOp>(printf);
  rewriter.create<LLVM::CallOp>(loc, printfFuncOp, operands);
}

/// NOTE: stolen directly from the MLIR LLVM lowering examples... this stuff
/// isn't exposed any where in public libraries!
/// Return a value representing an access into a global string with the given
/// name, creating the string if necessary.
static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                     StringRef name, StringRef value,
                                     ModuleOp module) {
  // Create the global at the entry of the module.
  LLVM::GlobalOp global;
  if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto type = LLVM::LLVMArrayType::get(
        IntegerType::get(builder.getContext(), 8), value.size());
    global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                            LLVM::Linkage::Internal, name,
                                            builder.getStringAttr(value),
                                            /*alignment=*/0);
  }

  // Get the pointer to the first character in the global string.
  Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
  Value cst0 = builder.create<LLVM::ConstantOp>(
      loc, IntegerType::get(builder.getContext(), 64),
      builder.getIntegerAttr(builder.getIndexType(), 0));
  return builder.create<LLVM::GEPOp>(
      loc,
      LLVM::LLVMPointerType::get(IntegerType::get(builder.getContext(), 8)),
      globalPtr, ArrayRef<Value>({cst0, cst0}));
}

static void insertIntegerLikeComparison(Location loc, ModuleOp module,
                                        PatternRewriter &rewriter, Value a,
                                        Value b) {
  auto cmp =
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, a, b);
  auto ifOp = rewriter.create<scf::IfOp>(loc, cmp);
  rewriter.setInsertionPointToStart(ifOp.getBody());

  llvm::SmallVector<Value> printfArgs;
  printfArgs.push_back(getOrCreateGlobalString(
      loc, rewriter, "cosimIntCmpErrStr", "COSIM: %d != %d", module));
  printfArgs.push_back(a);
  printfArgs.push_back(b);
  insertPrintfCall(loc, module, rewriter, printfArgs);
}

struct ConvertCompareIntegerLike : OpRewritePattern<cosim::CompareOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cosim::CompareOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.type().isIntOrIndex())
      return failure();

    insertIntegerLikeComparison(op.getLoc(), op->getParentOfType<ModuleOp>(),
                                rewriter, op.getRef(), op.getTarget());
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertCompareMemref : OpRewritePattern<cosim::CompareOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cosim::CompareOp op,
                                PatternRewriter &rewriter) const override {
    MemRefType memrefType = op.type().dyn_cast<MemRefType>();
    if (!memrefType)
      return failure();

    // Create a loop wherein we load each value in each memory, and compare
    // them.
    auto zero = rewriter.create<arith::ConstantOp>(op.getLoc(),
                                                   rewriter.getIndexAttr(0));
    auto one = rewriter.create<arith::ConstantOp>(op.getLoc(),
                                                  rewriter.getIndexAttr(1));

    llvm::SmallVector<Value> indices;
    scf::ForOp innerLoop = nullptr;
    for (auto dim : memrefType.getShape()) {
      auto ub = rewriter.create<arith::ConstantOp>(op.getLoc(),
                                                   rewriter.getIndexAttr(dim));
      innerLoop = rewriter.create<scf::ForOp>(op.getLoc(), zero, ub, one);
      indices.push_back(innerLoop.getInductionVar());
      rewriter.setInsertionPointToStart(innerLoop.getBody());
    }

    // Load values from the two compared memories
    auto loadRef =
        rewriter.create<memref::LoadOp>(op.getLoc(), op.getRef(), indices);
    auto targetRef =
        rewriter.create<memref::LoadOp>(op.getLoc(), op.getTarget(), indices);

    // Insert integer comparison
    insertIntegerLikeComparison(op.getLoc(), op->getParentOfType<ModuleOp>(),
                                rewriter, loadRef, targetRef);
    rewriter.eraseOp(op);
    return success();
  }
};

struct CosimLowerComparePass
    : public CosimLowerCompareBase<CosimLowerComparePass> {
  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<ConvertCompareIntegerLike, ConvertCompareMemref>(ctx);
    ConversionTarget target(*ctx);
    target.addIllegalOp<cosim::CallOp>();
    target.addIllegalOp<cosim::CompareOp>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<mlir::BuiltinDialect>();
    target.addLegalDialect<mlir::cf::ControlFlowDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<cosim::CosimDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<func::FuncDialect>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

struct ConvertStdCallPattern : OpRewritePattern<mlir::func::CallOp> {
  ConvertStdCallPattern(MLIRContext *ctx, StringRef from, StringRef ref,
                        const std::vector<std::string> &targets)
      : OpRewritePattern(ctx), from(from), ref(ref), targets(targets) {}
  LogicalResult matchAndRewrite(mlir::func::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getCallee() != from)
      return failure();

    rewriter.replaceOpWithNewOp<cosim::CallOp>(op, op.getCalleeAttr(),
                                               op.getResultTypes(), ref,
                                               targets, op.getOperands());
    return success();
  }

  StringRef from;
  StringRef ref;
  const std::vector<std::string> &targets;
};

struct CosimConvertCallPass
    : public CosimConvertCallBase<CosimConvertCallPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    auto ctx = funcOp.getContext();

    RewritePatternSet patterns(ctx);
    patterns.insert<ConvertStdCallPattern>(ctx, from, ref, targets);
    ConversionTarget target(*ctx);
    target.addDynamicallyLegalOp<mlir::func::CallOp>(
        [&](mlir::func::CallOp callOp) { return callOp.getCallee() != from; });
    target.addLegalDialect<cosim::CosimDialect>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> circt_hls::cosim::createCosimLowerCallPass() {
  return std::make_unique<CosimLowerCallPass>();
}

std::unique_ptr<mlir::Pass> circt_hls::cosim::createCosimLowerComparePass() {
  return std::make_unique<CosimLowerComparePass>();
}

std::unique_ptr<mlir::Pass> circt_hls::cosim::createCosimConvertCallPass() {
  return std::make_unique<CosimConvertCallPass>();
}
