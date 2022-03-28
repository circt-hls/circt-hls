//===- CalyxVerilatorWrapper.cpp - Calyx Verilator wrapper --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the CalyxVerilatorWrapper class,
// an HLT wrapper for wrapping Calyx.funcOp based kernels, simulated by
// Verilator.
//
//===----------------------------------------------------------------------===//

#include "circt-hls/Tools/hlt/WrapGen/calyx/CalyxVerilatorWrapper.h"
#include "circt-hls/Tools/hlt/WrapGen/VerilatorEmitterUtils.h"

using namespace llvm;
using namespace mlir;
using namespace circt;

namespace circt_hls {

struct GroundPort : public PortMapping {
  GroundPort(calyx::ComponentOp component, unsigned idx, bool isInput)
      : PortMapping(component), idx(idx), isInput(isInput) {}
  LogicalResult emit(raw_indented_ostream &osi) override {
    if (isInput) {
      auto ipi = comp.getInputPortInfo()[idx];
      auto inputName = ipi.name.str();
      osi << "addInputPort<CalyxInPort<";
      if (failed(emitVerilatorType(osi)))
        return failure();
      osi << ">(" << dutRef() << ");\n";
    } else {
      auto opi = comp.getOutputPortInfo()[idx];
      auto outputName = opi.name.str();
      osi << "addOutputPort<CalyxOutPort<";
      if (failed(emitVerilatorType(osi)))
        return failure();
      osi << ">>(" << dutRef() << ");\n";
    }

    return success();
  }

  LogicalResult emitVerilatorType(raw_ostream &osi) {
    if (isInput) {
      auto ipi = comp.getInputPortInfo()[idx];
      return emitVerilatorTypeFromWidth(osi, comp.getLoc(),
                                        ipi.type.getIntOrFloatBitWidth());
    } else {
      auto opi = comp.getOutputPortInfo()[idx];
      return emitVerilatorTypeFromWidth(osi, comp.getLoc(),
                                        opi.type.getIntOrFloatBitWidth());
    }
  }

  std::string dutRef() {
    if (isInput)
      return "&dut->" + comp.getInputPortInfo()[idx].name.str();
    else
      return "&dut->" + comp.getOutputPortInfo()[idx].name.str();
  }
  unsigned idx;
  bool isInput;
};

struct MemoryPort : public PortMapping {
  MemoryPort(calyx::ComponentOp component, MemRefType memref)
      : PortMapping(component), memref(memref) {}
  LogicalResult emit(raw_indented_ostream &osi) override {
    osi << "addInputPort<CalyxMemoryInterface<";
    if (emitVerilatorType(osi, comp.getLoc(), memref.getElementType()).failed())
      return failure();
    osi << ", ";
    // @todo: this doesn't really work for multidimensional memories
    if (failed(addrSignals.front().emitVerilatorType(osi)))
      return failure();
    osi << ">>(/*size=*/" << memref.getNumElements() << ",\n";
    osi.indent();

    auto emitSignal = [&](GroundPort &port, StringRef comment,
                          bool isInput) -> raw_indented_ostream & {
      osi << "/*" << comment << "=*/ std::make_shared<Calyx";
      osi << (isInput ? "In" : "Out");
      osi << "Port<";
      port.emitVerilatorType(osi);
      osi << ">>(";
      osi << port.dutRef() << ")";
      return osi;
    };

    emitSignal(*rdDataSignal, "rdDataSignal", false) << ",\n";
    emitSignal(*doneSignal, "doneSignal", false) << ",\n";
    emitSignal(*wrDataSignal, "wrDataSignal", true) << ",\n";
    emitSignal(*wrEnSignal, "wrEnSignal", true) << ",\n";
    for (auto it : llvm::enumerate(addrSignals)) {
      emitSignal(it.value(), "addrSignal" + std::to_string(it.index()), true);
      if (it.index() < addrSignals.size() - 1)
        osi << ",";
      osi << "\n";
    }
    osi.unindent();
    osi << ");";
    return success();
  }
  std::unique_ptr<GroundPort> rdDataSignal;
  std::unique_ptr<GroundPort> doneSignal;
  std::unique_ptr<GroundPort> wrDataSignal;
  std::unique_ptr<GroundPort> wrEnSignal;
  llvm::SmallVector<GroundPort> addrSignals;
  MemRefType memref;
};

LogicalResult CalyxVerilatorWrapper::init(Operation *refOp,
                                          Operation *kernelOp) {
  if (!(refOp))
    return refOp->emitError()
           << "Expected both a reference and a kernel operation for wrapping a "
              "Calyx simulator.";

  compOp = dyn_cast<calyx::ComponentOp>(refOp);
  if (!compOp)
    return refOp->emitOpError()
           << "expected reference operation to be a calyx.component operation.";

  generateIOPortMapping();

  return success();
}

void CalyxVerilatorWrapper::generateIOPortMapping() {
  // Counter for maintaing the current index within the in- or output port list
  // of the calyx component.
  unsigned calyxInPortIdx = 0;
  unsigned calyxOutPortIdx = 0;
  // Input ports
  for (auto it : llvm::enumerate(funcOp.getArguments())) {
    if (auto memrefType = it.value().getType().dyn_cast<MemRefType>()) {
      inputMapping[it.index()] =
          std::make_unique<MemoryPort>(compOp, memrefType);
      auto *memoryPort =
          static_cast<MemoryPort *>(inputMapping.at(it.index()).get());
      memoryPort->rdDataSignal =
          std::make_unique<GroundPort>(compOp, calyxInPortIdx++, true);
      memoryPort->doneSignal =
          std::make_unique<GroundPort>(compOp, calyxInPortIdx++, true);
      memoryPort->wrDataSignal =
          std::make_unique<GroundPort>(compOp, calyxOutPortIdx++, false);
      for (auto shape : memrefType.getShape())
        memoryPort->addrSignals.push_back(
            GroundPort(compOp, calyxOutPortIdx++, false));
      memoryPort->wrEnSignal =
          std::make_unique<GroundPort>(compOp, calyxOutPortIdx++, false);
    } else {
      // "normal" port
      inputMapping[it.index()] =
          std::make_unique<GroundPort>(compOp, calyxOutPortIdx++, false);
    }
  }

  // Output ports
  for (auto it : llvm::enumerate(funcOp.getResultTypes())) {
    assert(!it.value().isa<MemRefType>() &&
           "Memory ports only supported as input arguments");
    outputMapping[it.index()] =
        std::make_unique<GroundPort>(compOp, calyxOutPortIdx++, false);
  }
}

SmallVector<std::string> CalyxVerilatorWrapper::getIncludes() {
  SmallVector<std::string> includes;
  includes.push_back(("V" + funcName() + ".h").str());
  includes.push_back("circt-hls/Tools/hlt/Simulator/CalyxSimInterface.h");
  includes.push_back("circt-hls/Tools/hlt/Simulator/SimDriver.h");
  includes.push_back("cstdint");
  return includes;
}

LogicalResult CalyxVerilatorWrapper::emitPreamble(Operation *kernelOp) {
  if (emitIOTypes(emitVerilatorType).failed())
    return failure();

  // Emit model type.
  osi() << "using TModel = V" << funcName() << ";\n";
  osi() << "using " << funcName()
        << "SimInterface = CalyxSimInterface<TInput, TOutput, "
           "TModel>;\n\n";

  // Emit simulator.
  if (emitSimulator().failed())
    return failure();

  // Emit simulator driver type.
  osi() << "using TSim = " << funcName() << "Sim;\n";
  return success();
}

std::string CalyxVerilatorWrapper::getResName(unsigned idx) {
  return compOp.getOutputPortInfo()[idx].name.str();
}
std::string CalyxVerilatorWrapper::getInputName(unsigned idx) {
  return compOp.getInputPortInfo()[idx].name.str();
}

LogicalResult CalyxVerilatorWrapper::emitSimulator() {
  osi() << "class " << funcName() << "Sim : public " << funcName()
        << "SimInterface {\n";
  osi() << "public:\n";
  osi().indent();

  osi() << funcName() << "Sim() : " << funcName() << "SimInterface() {\n";
  osi().indent();

  osi() << "// --- Generic Verilator interface\n";
  osi() << "interface.clock = &dut->clk;\n";
  osi() << "interface.reset = &dut->reset;\n\n";

  auto outCtrlName = getResName(funcOp.getNumResults());
  osi() << "// --- Calyx interface\n";
  osi() << "go = std::make_shared<CalyxInPort<CData>>(&dut->go);\n";
  osi() << "done = std::make_shared<CalyxInPort<CData>>(&dut->done);";
  osi() << "\n\n";

  // We expect equivalence between the order of function arguments and the ports
  // of the Calyx component.

  osi() << "// --- Software interface\n";
  osi() << "// - Input ports\n";
  for (auto &it : inputMapping)
    if (failed(it.second->emit(osi())))
      return failure();

  osi() << "\n\n// - Output ports\n";
  for (auto &it : outputMapping)
    if (failed(it.second->emit(osi())))
      return failure();

  osi().unindent();
  osi() << "};\n";

  osi().unindent();
  osi() << "};\n\n";
  return success();
}

} // namespace circt_hls
