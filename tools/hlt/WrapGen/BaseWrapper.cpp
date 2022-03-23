//===- BaseWrapper.cpp - Simulation wrapper base class --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements BaseWrapper, the base class for all HLT wrappers.
//
//===----------------------------------------------------------------------===//

#include "circt-hls/Tools/hlt/WrapGen/BaseWrapper.h"
#include "circt-hls/Tools/hlt/WrapGen/CEmitterUtils.h"

using namespace llvm;
using namespace mlir;

namespace circt_hls {

LogicalResult BaseWrapper::wrap(mlir::func::FuncOp _funcOp, Operation *refOp,
                                Operation *kernelOp) {
  funcOp = _funcOp;
  if (init(refOp, kernelOp).failed())
    return failure();

  if (createFile(refOp->getLoc(), funcOp.getName() + ".cpp").failed())
    return failure();
  osi() << "// This file is generated. Do not modify!\n";

  // Emit includes
  for (auto include : getIncludes())
    osi() << "#include \"" << include << "\"\n";
  osi() << "\n";

  // Emit namespaces
  for (auto ns : getNamespaces())
    osi() << "using namespace " << ns << ";\n";
  osi() << "\n";

  // Emit preamble;
  if (emitPreamble(kernelOp).failed())
    return failure();

  // Emit simulator driver and instantiation. This is dependent on types TInput,
  // TOutput, TSim that should have been defined in emitPreamble.
  osi() << "using TSimDriver = SimDriver<TInput, TOutput, TSim>;\n";
  osi() << "static TSimDriver *driver = nullptr;\n\n";
  osi() << "void init_sim() {\n";
  osi() << "  assert(driver == nullptr && \"Simulator already initialized "
           "!\");\n";
  osi() << "  driver = new TSimDriver();\n";
  osi() << "}\n\n";

  // Emit async call
  llvm::raw_string_ostream callSigStream(callSignature);
  callSigStream << "extern \"C\" void " << funcOp.getName().str() + "_call"
                << "(";

  // InterleaveComma doesn't accept enumerate(inTypes)
  int i = 0;
  bool failed = false;
  interleaveComma(
      funcOp.getFunctionType().getInputs(), callSigStream, [&](auto inType) {
        auto varName = "in" + std::to_string(i++);
        failed |= emitType(callSigStream, _funcOp->getLoc(), inType, {varName})
                      .failed();
      });
  if (failed)
    return failure();
  callSigStream << ")";
  os() << callSignature << "{\n";
  osi().indent();
  emitAsyncCall();
  osi().unindent();
  osi() << "}\n\n";

  // Emit async await
  llvm::raw_string_ostream awaitSigStream(awaitSignature);
  awaitSigStream << "extern \"C\" ";
  if (emitTypes(awaitSigStream, funcOp.getLoc(),
                funcOp.getFunctionType().getResults())
          .failed())
    return failure();
  awaitSigStream << " " << funcOp.getName().str() + "_await"
                 << "()";
  os() << awaitSignature << "{\n";
  osi().indent();
  emitAsyncAwait();

  // End
  osi().unindent();
  osi() << "}\n";

  // Create wrapper header file
  if (createFile(refOp->getLoc(), funcOp.getName() + ".h").failed())
    return failure();
  osi() << "// This file is generated. Do not modify!\n";
  // cstdint should be included to support the int#_t types used in the function
  // arguments.
  osi() << "#include \"cstdint\"\n";
  osi() << callSignature << ";\n";
  osi() << awaitSignature << ";\n";

  return success();
}

LogicalResult BaseWrapper::emitIOTypes(const TypeEmitter &emitter) {
  auto funcType = funcOp.getFunctionType();

  // Emit in types.
  for (auto &inType : enumerate(funcType.getInputs())) {
    osi() << "using TArg" << inType.index() << " = ";
    if (emitter(osi(), funcOp.getLoc(), inType.value(), {}).failed())
      return failure();
    osi() << ";\n";
  }
  osi() << "using TInput = std::tuple<";
  interleaveComma(llvm::iota_range(0U, funcOp.getNumArguments(), false), osi(),
                  [&](unsigned i) { osi() << "TArg" << i; });
  osi() << ">;\n\n";

  // Emit out types.
  for (auto &outType : enumerate(funcType.getResults())) {
    osi() << "using TRes" << outType.index() << " = ";
    if (emitter(osi(), funcOp.getLoc(), outType.value(), {}).failed())
      return failure();
    osi() << ";\n";
  }
  osi() << "using TOutput = std::tuple<";
  interleaveComma(llvm::iota_range(0U, funcType.getNumResults(), false), osi(),
                  [&](unsigned i) { osi() << "TRes" << i; });
  osi() << ">;\n\n";
  return success();
}

LogicalResult BaseWrapper::createFile(Location loc, Twine fn) {
  std::error_code EC;
  SmallString<128> absFn = outDir;
  sys::path::append(absFn, fn);
  outputFile = std::make_unique<raw_fd_ostream>(absFn, EC, sys::fs::OF_None);
  if (EC)
    return emitError(loc) << "Error while opening file";
  outputFilename = absFn.str().str();
  outputFileIndented = std::make_unique<raw_indented_ostream>(*outputFile);
  return success();
}

void BaseWrapper::emitAsyncCall() {
  osi() << "if (driver == nullptr)\n";
  osi() << "  init_sim();\n";

  // Pack arguments
  osi() << "TInput input;\n";
  for (auto arg : llvm::enumerate(funcOp.getFunctionType().getInputs())) {
    // Reinterpret/static cast here is just a hack around software interface
    // providing i.e. int32_t* as pointer type, and verilator using uint32_t*.
    // Should obviously be fixed so we don't throw away type safety.
    bool isPtr = arg.value().isa<MemRefType>();

    osi() << "std::get<" << arg.index() << ">(input) = ";
    osi() << (isPtr ? "reinterpret_cast" : "static_cast");
    osi() << "<TArg" << arg.index() << ">(in" << arg.index() << ");\n";
  }

  // Push to driver
  osi() << "driver->push(input); // non-blocking\n";
}

void BaseWrapper::emitAsyncAwait() {
  osi() << "TOutput output = driver->pop(); // blocking\n";
  switch (funcOp.getNumResults()) {
  case 0: {
    osi() << "return;\n";
    break;
  }
  case 1: {
    osi() << "return std::get<0>(output);\n";
    break;
  }
  default: {
    osi() << "return output;\n";
    break;
  }
  }
}

} // namespace circt_hls
