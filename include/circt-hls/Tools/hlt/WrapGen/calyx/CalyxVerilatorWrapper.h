//===- CalyxVerilatorWrapper.h - Calyx Verilator wrapper ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definition of the CalyxVerilatorWrapper class, an
// HLT wrapper for wrapping Calyx based kernels simulated by Verilator.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_HLT_WRAPGEN_CALYX_CALYXVERILATORWRAPPER_H
#define CIRCT_TOOLS_HLT_WRAPGEN_CALYX_CALYXVERILATORWRAPPER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/Calyx/CalyxOps.h"

#include "circt-hls/Tools/hlt/WrapGen/BaseWrapper.h"
#include "circt-hls/Tools/hlt/WrapGen/CEmitterUtils.h"
#include "circt-hls/Tools/hlt/WrapGen/VerilatorEmitterUtils.h"

using namespace mlir;
using namespace circt;

struct PortMapping {
  PortMapping(calyx::ComponentOp component) : comp(component)  {}
  virtual ~PortMapping() = default;
  virtual LogicalResult emit(raw_indented_ostream& osi) = 0;
  calyx::ComponentOp comp;
};

namespace circt_hls {

class CalyxVerilatorWrapper : public BaseWrapper {
public:
  using BaseWrapper::BaseWrapper;
  LogicalResult init(Operation *refOp, Operation *kernelOp) override;
  LogicalResult emitPreamble(Operation *kernelOp) override;

protected:
  SmallVector<std::string> getIncludes() override;
  SmallVector<std::string> getNamespaces() override { return {"circt", "hlt"}; }

private:
  LogicalResult emitSimulator();
  LogicalResult emitInputPort(Type t, unsigned idx);
  LogicalResult emitOutputPort(Type t, unsigned idx);
  LogicalResult emitExtMemPort(MemRefType t, unsigned idx);

  // Returns the port names for the respective in- or output index.
  std::string getResName(unsigned idx);
  std::string getInputName(unsigned idx);

  // Operations representing the reference and firrtl modules of the kernel.
  calyx::ComponentOp compOp;

  // A mapping between a function argument/result index and its resulting hardware ports.
  std::map<unsigned, std::unique_ptr<PortMapping>> inputMapping;
  std::map<unsigned, std::unique_ptr<PortMapping>> outputMapping;
  void generateIOPortMapping();
};

} // namespace circt_hls

#endif // CIRCT_TOOLS_HLT_WRAPGEN_CALYX_CALYXVERILATORWRAPPER_H
