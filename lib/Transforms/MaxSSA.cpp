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
using namespace func;
using namespace circt_hls;

using ValueFilterCallbackFn = llvm::function_ref<bool(Value)>;

/// Rewrites the terminator in 'from' to pass an additional argument 'v' when
/// passing control flow to 'to'.
static LogicalResult rewriteControlFlowToBlock(Block *from, Block *to,
                                               Value v) {
  auto termOp = from->getTerminator();
  auto branchOp = dyn_cast<BranchOpInterface>(termOp);
  if (!branchOp)
    return termOp->emitOpError() << "expected terminator op within control "
                                    "flow to be a branch-like op";

  // Find 'to' successor in the branch op.
  auto successors = branchOp->getSuccessors();
  auto succIt = llvm::find(successors, to);
  assert(succIt != successors.end());
  unsigned succIdx = std::distance(successors.begin(), succIt);

  // Add the value 'v' as a block argument.
  branchOp.getMutableSuccessorOperands(succIdx)->append(v);
  return success();
}

/// Rewrites uses of 'oldV' in 'b' to 'newV'.
static void rewriteUsageInBlock(Block *b, Value oldV, Value newV) {
  for (auto &use : llvm::make_early_inc_range(oldV.getUses()))
    if (use.getOwner()->getBlock() == b) {
      use.set(newV);
    }
}

/// Perform a depth-first search backwards through the CFG graph of a program,
/// starting from 'use', add a new block argument of type(v) to the block and
/// replaces all uses of 'v' with the new block argument.
///
/// Arguments: 'use': ablock where a value v flows through. 'succ': a successor
/// block of the 'use' block. Notably this conversion backtracks through the BB
/// CFG graph, so 'succ' will be a basic block that called backtrackAndConvert
/// on 'use'. 'inBlockValues': A mapping containing the appended block argument
/// when backtracking through a basic block. 'convertedControlFlow': A mapping
/// containing, for a given block (key) which successor operand range in the
/// terminator have been rewritten to the new block argument signature.
static LogicalResult
backtrackAndConvert(Block *use, Block *succ, Value v,
                    DenseMap<Value, DenseMap<Block *, Value>> &inBlockValues,
                    DenseMap<Value, DenseMap<Block *, DenseSet<Block *>>>
                        &convertedControlFlow) {
  // The base case is when we've backtracked to the Block which defines the
  // value. In these cases, set the actual value as the converted value.
  if (v.getParentBlock() == use)
    inBlockValues[v][use] = v;

  if (inBlockValues[v].count(use) == 0) {

    // Rewrite this blocks' block arguments to take in a new value of 'v' type.
    use->addArgument(v.getType(), v.getLoc());
    Value newBarg = use->getArguments().back();

    // Register the converted block argument in case other branches in the CFG
    // arrive here later.
    inBlockValues[v][use] = newBarg;
    rewriteUsageInBlock(use, v, newBarg);

    // Recurse through the predecessors of this block.
    for (auto pred : use->getPredecessors())
      if (failed(backtrackAndConvert(pred, use, v, inBlockValues,
                                     convertedControlFlow)))
        return failure();
  }

  // Rewrite control flow to the 'succ' block through the terminator, if not
  // already done.
  if (succ && convertedControlFlow[v][use].count(succ) == 0) {
    auto alreadyinBlockValues = inBlockValues[v].find(use);
    assert(alreadyinBlockValues != inBlockValues[v].end());
    if (failed(
            rewriteControlFlowToBlock(use, succ, alreadyinBlockValues->second)))
      return failure();
    convertedControlFlow[v][use].insert(succ);
  }

  return success();
}

namespace {

struct MaxSSAFormConverter {
public:
  /// An optional filterFn may be provided to dynamically filter out values
  /// from being converted.
  MaxSSAFormConverter(ValueFilterCallbackFn filterFn = nullptr)
      : filterFn(filterFn) {}

  LogicalResult convertFunction(FuncOp function) {
    auto walkRes = function.walk([&](Operation *op) {
      // Run on operation results.
      SetVector<Value> visited;
      if (llvm::any_of(op->getResults(), [&](Value res) {
            visited.insert(res);
            return failed(runOnValue(res));
          })) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    // Run on block arguments.
    for (auto &block : function) {
      if (llvm::any_of(block.getArguments(),
                       [&](Value barg) { return failed(runOnValue(barg)); }))
        return failure();
    }

    if (walkRes.wasInterrupted())
      return failure();

    assert(
        succeeded(verifyFunction(function)) &&
        "Some values were still referenced outside of their defining block!");

    return success();
  }

  LogicalResult convertValue(Value v) {
    auto *defOp = v.getDefiningOp();
    auto funcOp = dyn_cast<FuncOp>(defOp->getParentOp());
    if (!funcOp)
      return defOp->emitOpError() << "Expected parent operation to be a "
                                     "function, but got "
                                  << defOp->getParentOp()->getName();

    return runOnValue(v);
  }

private:
  /// Verifies that all values which are not filtered indeed are only referenced
  /// within their defining block.
  LogicalResult verifyFunction(FuncOp f) const;

  /// Driver which will run backtrackAndConvert on values referenced outside
  /// their defining block. Returns failure in case the pass failed to apply.
  /// This may happen when nested regions exist within the FuncOp which this
  /// pass is applied to, or if non branch-like control flow is used.
  LogicalResult runOnValue(Value v);

  /// A mapping {original value : {block : replaced value}} representing
  /// 'original value' has been replaced in 'block' with 'replaced value'".
  DenseMap<Value, DenseMap<Block *, Value>> inBlockValues;

  /// A mapping {original value : {block : succ block}} representing
  /// 'original value' has already been passed from 'block' to 'succ block'
  /// through the terminator of 'block'.
  DenseMap<Value, DenseMap<Block *, DenseSet<Block *>>> convertedControlFlow;

  /// An optional filter function to dynamically determine whether a value
  /// should be considered for SSA maximization.
  ValueFilterCallbackFn filterFn;
};

LogicalResult MaxSSAFormConverter::verifyFunction(FuncOp f) const {
  for (auto &op : f.getOps()) {
    auto isValid = [&](Value v) {
      if (filterFn && filterFn(v))
        return true;
      return v.getParentBlock() == op.getBlock();
    };

    if (!llvm::all_of(op.getOperands(), isValid))
      return op.emitOpError()
             << "has operands that are not defined within its block";
  }
  return success();
}

LogicalResult MaxSSAFormConverter::runOnValue(Value v) {
  if (filterFn && filterFn(v))
    return success();
  Block *definingBlock = v.getParentBlock();

  SetVector<Block *> usedInBlocks;
  for (Operation *user : v.getUsers())
    usedInBlocks.insert(user->getBlock());

  for (Block *userBlock : usedInBlocks) {
    if (definingBlock != userBlock) {
      // This is a case of using an SSA value through basic block dominance.
      if (userBlock->getParent() != definingBlock->getParent())
        return emitError(v.getLoc()) << "can only convert SSA usage across "
                                        "blocks in the same region.";

      if (failed(backtrackAndConvert(userBlock, /*succ=*/nullptr, v,
                                     inBlockValues, convertedControlFlow)))
        return failure();
    }
  }
  return success();
}

} // namespace

LogicalResult convertToMaximalSSA(FuncOp func,
                                  ValueFilterCallbackFn filterFn = nullptr) {
  return MaxSSAFormConverter(filterFn).convertFunction(func);
}

LogicalResult convertToMaximalSSA(Value value) {
  return MaxSSAFormConverter().convertValue(value);
}

struct MaxSSAFormPass : public MaxSSAFormBase<MaxSSAFormPass> {
public:
  void runOnOperation() override {
    FuncOp func = getOperation();

    // Do not modify memref's
    ValueFilterCallbackFn filterFn = nullptr;

    if (ignoreMemref)
      filterFn = [&](Value v) { return v.getType().isa<MemRefType>(); };

    if (convertToMaximalSSA(func, filterFn).failed())
      return signalPassFailure();
  }
};

namespace circt_hls {
std::unique_ptr<mlir::Pass> createMaxSSAFormPass() {
  return std::make_unique<MaxSSAFormPass>();
}
} // namespace circt_hls
