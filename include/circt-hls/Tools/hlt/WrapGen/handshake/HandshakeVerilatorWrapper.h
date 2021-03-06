//===- HandshakeVerilatorWrapper.h - Handshake Verilator wrapper ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definition of the HandshakeVerilatorWrapper class, an
// HLT wrapper for wrapping handshake.funcop based kernels simulated by
// Verilator.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_HLT_WRAPGEN_HANDSHAKE_HANDSHAKEVERILATORWRAPPER_H
#define CIRCT_TOOLS_HLT_WRAPGEN_HANDSHAKE_HANDSHAKEVERILATORWRAPPER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"

#include "circt-hls/Tools/hlt/WrapGen/BaseWrapper.h"
#include "circt-hls/Tools/hlt/WrapGen/CEmitterUtils.h"
#include "circt-hls/Tools/hlt/WrapGen/VerilatorEmitterUtils.h"

using namespace mlir;
using namespace circt;

namespace circt_hls {

class HandshakeVerilatorWrapper : public BaseWrapper {
public:
  using BaseWrapper::BaseWrapper;
  LogicalResult init(Operation *refOp, Operation *kernelOp) override;
  LogicalResult emitPreamble(Operation *kernelOp) override;

protected:
  SmallVector<std::string> getIncludes() override;
  SmallVector<std::string> getNamespaces() override { return {"circt", "hlt"}; }

private:
  // Returns the index in the firrtl port argument list of the input control
  // port.
  unsigned inCtrlIdx();

  LogicalResult emitSimulator();
  LogicalResult emitInputPort(Type t, unsigned idx);
  LogicalResult emitOutputPort(Type t, unsigned idx);
  LogicalResult emitExtMemPort(MemRefType t, unsigned idx);

  // Returns the port names for the respective in- or output index.
  std::string getResName(unsigned idx);
  std::string getInputName(unsigned idx);

  // Returns the data width for a port. The port is expected to be a firrtl bundle.
  unsigned getBundleDataWidth(firrtl::BundleType idx);

  // Operations representing the reference and firrtl modules of the kernel.
  handshake::FuncOp hsOp;
  firrtl::FModuleLike firrtlOp;
};

} // namespace circt_hls

#endif // CIRCT_TOOLS_HLT_WRAPGEN_HANDSHAKE_HANDSHAKEVERILATORWRAPPER_H
