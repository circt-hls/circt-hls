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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/TypeSwitch.h"

#include <set>

using namespace mlir;
using namespace circt_hls;

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
  for (auto &use : oldV.getUses())
    if (use.getOwner()->getBlock() == b)
      use.set(newV);
}

/// Perform a depth-first search backwards through the CFG graph of a program,
/// starting from 'use', add a new block argument of type(v) to the block and
/// replaces all uses of 'v' with the new block argument.
///
/// Arguments: 'use': ablock where a value v flows through. 'succ': a successor
/// block of the 'use' block. Notably this conversion backtracks through the BB
/// CFG graph, so 'succ' will be a basic block that called backtrackAndConvert
/// on 'use'. 'convertedBArgs': A mapping containing the appended block argument
/// when backtracking through a basic block. 'convertedControlFlow': A mapping
/// containing, for a given block (key) which successor opernad range in the
/// terminator have been rewritten to the new block argument signature.
static LogicalResult backtrackAndConvert(
    Block *use, Block *succ, Value v, DenseMap<Block *, Value> &convertedBArgs,
    DenseMap<Block *, std::set<Block *>> &convertedControlFlow) {
  // The base case is when we've backtracked to the Block which defines the
  // value. In these cases, set the actual value as the converted value.
  if (v.getParentBlock() == use)
    convertedBArgs[use] = v;

  auto alreadyConvertedBArgs = convertedBArgs.find(use);
  if (alreadyConvertedBArgs == convertedBArgs.end()) {
    // Rwrite this blocks' block arguments to take in a new value of 'v' type
    use->addArgument(v.getType());
    Value newBarg = use->getArguments().back();

    // Register the converted block argument in case other branches in the CFG
    // arrive here earlier.
    convertedBArgs[use] = newBarg;
    rewriteUsageInBlock(use, v, newBarg);

    // Recurse through the predecessors of this block
    for (auto pred : use->getPredecessors())
      if (backtrackAndConvert(pred, use, v, convertedBArgs,
                              convertedControlFlow)
              .failed())
        return failure();
  }

  // Rewrite control flow to the succ block through the termiantor, if not
  // already done.
  if (succ && convertedControlFlow[use].count(succ) == 0) {
    alreadyConvertedBArgs = convertedBArgs.find(use);
    assert(alreadyConvertedBArgs != convertedBArgs.end());
    if (rewriteControlFlowToBlock(use, succ, alreadyConvertedBArgs->second)
            .failed())
      return failure();
    convertedControlFlow[use].insert(succ);
  }

  return success();
}

namespace {

struct MaxSSAFormPass : public MaxSSAFormBase<MaxSSAFormPass> {
public:
  void runOnFunction() override {
    FuncOp function = getOperation();

    function.walk([&](Operation *op) {
      if (llvm::any_of(op->getOperands(), [&](Value operand) {
            return runOnValue(operand).failed();
          })) {
        signalPassFailure();
        return;
      }
    });

    assert(
        verifyFunction(function).succeeded() &&
        "Some values were still referenced outside of their defining block!");
  }

private:
  /// Returns true if this value is ignored in SSA maximisation.
  bool isIgnored(Value v) const;

  /// Verifies that all values indeed are only referenced within their defining
  /// block.
  LogicalResult verifyFunction(FuncOp f) const;

  /// Driver which will run backtrackAndConvert on values referenced outside
  /// their defining block.
  LogicalResult runOnValue(Value v);
};

/// Returns true if this value is ignored in SSA maximisation.
bool MaxSSAFormPass::isIgnored(Value v) const {
  Type t = v.getType();

  return llvm::TypeSwitch<Type, bool>(t)
      .Case<MemRefType>([&](auto) { return static_cast<bool>(ignoreMemref); })
      .Default([&](auto) {
        return llvm::find(ignoredDialects, t.getDialect().getNamespace()) !=
               ignoredDialects.end();
      });
}

LogicalResult MaxSSAFormPass::verifyFunction(FuncOp f) const {
  for (auto &op : f.getOps()) {
    auto isValid = [&](Value v) {
      return isIgnored(v) || v.getParentBlock() == op.getBlock();
    };

    if (!llvm::all_of(op.getOperands(), isValid))
      return op.emitOpError()
             << "has operands that are not defined within its block";
  }
  return success();
}

LogicalResult MaxSSAFormPass::runOnValue(Value v) {
  if (isIgnored(v))
    return success();
  Block *definingBlock = v.getParentBlock();
  for (auto user : v.getUsers()) {
    Block *userBlock = user->getBlock();
    if (definingBlock != userBlock) {
      // This is a case of using an SSA value through basic block dominance.
      if (userBlock->getParent() != definingBlock->getParent())
        return user->emitOpError() << "can only convert SSA usage across "
                                      "blocks in the same region.";

      DenseMap<Block *, Value> convertedBArgs;
      DenseMap<Block *, std::set<Block *>> convertedControlFlow;
      if (backtrackAndConvert(userBlock, nullptr, v, convertedBArgs,
                              convertedControlFlow)
              .failed())
        return failure();
    }
  }
  return success();
}

} // namespace

namespace circt_hls {
std::unique_ptr<mlir::Pass> createMaxSSAFormPass() {
  return std::make_unique<MaxSSAFormPass>();
}
} // namespace circt_hls
