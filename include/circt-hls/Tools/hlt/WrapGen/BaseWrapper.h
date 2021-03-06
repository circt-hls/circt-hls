//===- BaseWrapper.h - Simulation wrapper base class ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines BaseWrapper, the base class for all HLT wrappers.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_HLS_TOOLS_HLT_WRAPGEN_BASEWRAPPER_H
#define CIRCT_HLS_TOOLS_HLT_WRAPGEN_BASEWRAPPER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

namespace circt_hls {

class BaseWrapper {
public:
  BaseWrapper(StringRef outDir) : outDir(outDir) {}
  virtual ~BaseWrapper(){};

  /// Initializes the subclassing wrapper. In here, we check for whether the
  /// types of the reference and kernel operations are as expected.
  virtual LogicalResult init(Operation *refOp, Operation *kernelOp) = 0;

  /// The wrap function takes a source operation and a kernel operation. The
  /// kernel operation is wrapper-specific, whereas the source operation
  /// represents the "software" interface that a kernel is expected to adhere
  /// to.
  LogicalResult wrap(mlir::func::FuncOp funcOp, Operation *refOp,
                     Operation *kernelOp);

  std::string getOutputFileName() { return outputFilename; }

protected:
  virtual SmallVector<std::string> getNamespaces() = 0;
  virtual SmallVector<std::string> getIncludes() = 0;

  virtual void emitAsyncCall();
  virtual void emitAsyncAwait();

  virtual LogicalResult emitPreamble(Operation * /*kernelOp*/) {
    return success();
  };
  StringRef funcName() { return funcOp.getName(); }
  using TypeEmitter = std::function<LogicalResult(llvm::raw_ostream &, Location,
                                                  Type, Optional<StringRef>)>;
  LogicalResult emitIOTypes(const TypeEmitter &emitter);

  llvm::raw_fd_ostream &os() { return *outputFile; }
  raw_indented_ostream &osi() { return *outputFileIndented; }

  std::unique_ptr<llvm::raw_fd_ostream> outputFile;
  std::unique_ptr<raw_indented_ostream> outputFileIndented;
  std::string outputFilename;
  StringRef outDir;
  func::FuncOp funcOp;

private:
  /// Creates an output file with filename fn, and associates os() and osi()
  /// with this file.
  LogicalResult createFile(Location loc, Twine fn);

  /// Function signatures of the call and await functions. These will be written
  /// to a separate header file.
  std::string callSignature, awaitSignature;
};

} // namespace circt_hls

#endif // CIRCT_TOOLS_HLT_WRAPGEN_BASEWRAPPER_H
