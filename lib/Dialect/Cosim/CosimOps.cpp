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
  if (!op.ref().isa<StringAttr>())
    return op.emitOpError() << "expected 'ref' to be a string.";

  if (llvm::any_of(op.targets(),
                   [&](Attribute attr) { return !attr.isa<StringAttr>(); }))
    return op.emitOpError() << "expected 'targets' to be an array of strings.";

  if (llvm::any_of(op.targets(), [&](Attribute attr) {
        return attr.cast<StringAttr>().strref() ==
               op.ref().cast<StringAttr>().strref();
      }))
    return op.emitOpError() << "do not include the reference function in the "
                               "set of target functions.";

  return success();
}

#define GET_OP_CLASSES
#include "circt-hls/Dialect/Cosim/Cosim.cpp.inc"
