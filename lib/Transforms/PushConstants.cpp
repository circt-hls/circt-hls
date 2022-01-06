//===- PushConstants.cpp - Constant pushing pass -----------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pushes constant definitions into the basic blocks where they are referenced.
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

namespace {

// We assume that the IR has been canonicalized beforehand, which has merged
// constants together.
struct PushConstantsPass : public PushConstantsBase<PushConstantsPass> {
public:
  void runOnFunction() override {
    auto f = getOperation();
    auto ctx = f.getContext();

    OpBuilder builder(ctx);

    for (auto constantOp :
         llvm::make_early_inc_range(f.getOps<arith::ConstantOp>())) {
      auto constantBlock = constantOp->getBlock();
      bool userInOwnBlock = false;
      // Maintain a mapping of the constant copy in each block, to avoid
      // multiple constant definitions in the same block.
      llvm::DenseMap<Block *, Operation *> copyInBlock;

      // llvm::make_early_inc_range doesn't seem to work, so we manually create
      // a set.
      llvm::SetVector<Operation *> users;
      for (auto user : constantOp->getUsers())
        users.insert(user);
      for (auto user : users) {
        auto userBlock = user->getBlock();
        if (userBlock == constantBlock) {
          userInOwnBlock = true;
          continue;
        }

        // Add a copy of the constant in the user block
        Operation *newConstantOp = nullptr;
        auto copy = copyInBlock.find(userBlock);
        if (copy != copyInBlock.end())
          newConstantOp = copy->second;
        else {
          builder.setInsertionPointToStart(userBlock);
          newConstantOp = builder.create<arith::ConstantOp>(
              constantOp.getLoc(), constantOp.getValue());
        }

        user->replaceUsesOfWith(constantOp, newConstantOp->getResult(0));
        copyInBlock[userBlock] = newConstantOp;
      }
      if (!userInOwnBlock)
        constantOp.erase();
    }
  }
};

} // namespace

namespace circt_hls {
std::unique_ptr<mlir::Pass> createPushConstantsPass() {
  return std::make_unique<PushConstantsPass>();
}
} // namespace circt_hls
