// A pass for removing any attributes created by unregistered dialects.

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

struct CleanUnregisteredAttrsPass
    : public CleanUnregisteredAttrsBase<CleanUnregisteredAttrsPass> {
public:
  void clean(Operation *op) {

    llvm::SmallVector<NamedAttribute, 4> cleanedAttrs;
    llvm::copy_if(op->getAttrs(), std::back_inserter(cleanedAttrs),
                  [&](NamedAttribute attr) {
                    auto attrName = attr.getName();

                    // Is it a registered dialect?
                    if (attr.getNameDialect())
                      return true;

                    // Split attribute name into dialect and attribute name.
                    auto dialectPrefix = attrName.strref().split('.').first;
                    return dialectPrefix != dialectName;
                  });

    op->setAttrs(cleanedAttrs);
    for (auto &reg : op->getRegions()) {
      for (auto &childOp : reg.getOps())
        clean(&childOp);
    }
  }

  void runOnOperation() override {
    if (dialectName.empty()) {
      emitError(getOperation().getLoc())
          << "Must specify some dialect prefix with --dialect";
      signalPassFailure();
      return;
    }

    clean(getOperation());
  }
};

} // namespace

namespace circt_hls {

std::unique_ptr<mlir::Pass> createCleanUnregisteredAttrsPass() {
  return std::make_unique<CleanUnregisteredAttrsPass>();
}
} // namespace circt_hls
