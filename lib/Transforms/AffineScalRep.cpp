//===- AffineScalRep.cpp - Affine scalar replacement ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the Affine scalar replacement pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Dominance.h"

using namespace mlir;
using namespace circt_hls;

namespace {
struct AffineScalRepPass : public AffineScalRepBase<AffineScalRepPass> {
public:
  void runOnOperation() override;

private:
  void forwardStoreToReturn(ReturnOp returnOp,
                            SmallVectorImpl<Operation *> &storeOpsToErase,
                            SmallPtrSetImpl<Value> &memrefsToErase,
                            DominanceInfo &domInfo);
};
} // namespace

/// Attempt to replace the operands to returnOp with values stored into memory
/// which is returned. This check involves three components: 1) The store and
/// return must be on the same location 2) The store must dominate (and
/// therefore must always occur prior to) the return 3) No other operations will
/// overwrite the memory loaded between the given store and return.  If such a
/// value exists, the replaced value will be used in the `returnOp` operands and
/// its memref will be added to `memrefsToErase`.
void AffineScalRepPass::forwardStoreToReturn(
    ReturnOp returnOp, SmallVectorImpl<Operation *> &storeOpsToErase,
    SmallPtrSetImpl<Value> &memrefsToErase, DominanceInfo &domInfo) {

  for (auto &opOperand : returnOp->getOpOperands()) {
    // The store op candidate for forwarding that satisfies all conditions
    // to replace the load, if any.
    Operation *lastWriteStoreOp = nullptr;

    MemRefType memRefType = opOperand.get().getType().dyn_cast<MemRefType>();
    if (!memRefType)
      continue;

    Operation *definingOp = opOperand.get().getDefiningOp();
    if (!definingOp)
      continue;

    for (auto *user : definingOp->getUsers()) {
      auto storeOp = dyn_cast<AffineWriteOpInterface>(user);
      if (!storeOp)
        continue;

      if (lastWriteStoreOp && !domInfo.dominates(lastWriteStoreOp, storeOp)) {
        storeOpsToErase.push_back(storeOp);
        continue;
      }

      // TODO: check intervening.

      lastWriteStoreOp = storeOp;
    }

    if (!lastWriteStoreOp)
      return;

    // Perform the actual store to load forwarding.
    Value storeVal =
        cast<AffineWriteOpInterface>(lastWriteStoreOp).getValueToStore();
    // Check if 2 values have the same shape. This is needed for affine vector
    // loads and stores.
    if (storeVal.getType() != memRefType.getElementType())
      return;

    // Record the store and memref for a later sweep to optimize away.
    storeOpsToErase.push_back(lastWriteStoreOp);
    memrefsToErase.insert(opOperand.get());

    // Update the operand to the stored value.
    opOperand.set(storeVal);

    // Update the function return type with the stored value's type.
    auto f = returnOp->getParentOfType<FuncOp>();
    auto argTypes = f.getType().getInputs();
    SmallVector<Type> resultTypes(f.getType().getResults().begin(),
                                  f.getType().getResults().end());
    resultTypes[opOperand.getOperandNumber()] = storeVal.getType();
    f.setType(FunctionType::get(f.getContext(), argTypes, resultTypes));
  }

  return;
}

void AffineScalRepPass::runOnOperation() {
  // Only supports single block functions at the moment.
  FuncOp f = getOperation();

  // Load op's whose results were replaced by those forwarded from stores.
  SmallVector<Operation *, 8> opsToErase;

  // A list of memref's that are potentially dead / could be eliminated.
  SmallPtrSet<Value, 4> memrefsToErase;

  auto &domInfo = getAnalysis<DominanceInfo>();

  // Perform store to return forwarding.
  auto returnOp = cast<ReturnOp>(f.getBody().front().getTerminator());
  forwardStoreToReturn(returnOp, opsToErase, memrefsToErase, domInfo);

  // Erase all store op's which don't impact the program
  for (auto *op : opsToErase)
    op->erase();

  // Check if the store fwd'ed memrefs are now left with only stores and can
  // thus be completely deleted. Note: the canonicalize pass should be able
  // to do this as well, but we'll do it here since we collected these anyway.
  for (auto memref : memrefsToErase) {
    memref.dump();
    // If the memref hasn't been alloc'ed in this function, skip.
    Operation *defOp = memref.getDefiningOp();
    if (!defOp || !isa<memref::AllocOp, memref::AllocaOp>(defOp))
      // TODO: if the memref was returned by a 'call' operation, we
      // could still erase it if the call had no side-effects.
      continue;
    if (llvm::any_of(memref.getUsers(), [&](Operation *ownerOp) {
          return !isa<AffineWriteOpInterface, memref::DeallocOp>(ownerOp);
        }))
      continue;

    // Erase all stores, the dealloc, and the alloc on the memref.
    for (auto *user : llvm::make_early_inc_range(memref.getUsers()))
      user->erase();
    defOp->erase();
  }
}

namespace circt_hls {
std::unique_ptr<mlir::Pass> createAffineScalRepPass() {
  return std::make_unique<AffineScalRepPass>();
}
} // namespace circt_hls
