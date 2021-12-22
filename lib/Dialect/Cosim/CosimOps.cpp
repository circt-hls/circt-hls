#include "circt-hls/Dialect/Cosim/CosimOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt_hls;
using namespace cosim;

static LogicalResult verifyCallOp(cosim::CallOp op) {
  if (llvm::any_of(op.targets(), [&](Attribute attr) {
        return attr.cast<StringAttr>().strref() == op.ref();
      }))
    return op.emitOpError() << "do not include the reference function in the "
                               "set of target functions.";

  return success();
}

void cosim::CallOp::build(OpBuilder &odsBuilder, OperationState &state,
                          SymbolRefAttr func, mlir::TypeRange results,
                          StringRef ref, ArrayRef<std::string> targets,
                          ValueRange operands) {
  auto ctx = odsBuilder.getContext();
  state.addOperands(operands);
  state.addAttribute("func", func);
  state.addTypes(results);
  state.addAttribute("ref", StringAttr::get(ref, ctx));

  llvm::SmallVector<Attribute> targetsAttr;
  for (auto &target : targets)
    targetsAttr.push_back(StringAttr::get(target, ctx));

  state.addAttribute("targets", ArrayAttr::get(ctx, targetsAttr));
}

#define GET_OP_CLASSES
#include "circt-hls/Dialect/Cosim/Cosim.cpp.inc"