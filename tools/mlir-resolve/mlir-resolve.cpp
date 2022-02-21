//===- mlir-resolve.cpp - Cross-file symbol resolution --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the mlir-resolve tool. The tool will, given two input
// files, resolve any private symbols of file 1 which are defined in file 2, and
// output the resulting module.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser.h"

#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"

#include <iostream>

using namespace llvm;
using namespace circt;
using namespace mlir;

static cl::opt<std::string> inputFile1("file1", cl::Required,
                                       cl::desc("<first file>"), cl::init("-"));
static cl::opt<std::string>
    inputFile2("file2", cl::Optional, cl::desc("<second file>"), cl::init("-"));

/// Container for the current set of loaded modules.
static SmallVector<mlir::OwningOpRef<mlir::ModuleOp>> modules;

/// Load a module from the argument file fn into the modules vector.
static ModuleOp getModule(MLIRContext *ctx, StringRef fn) {
  auto file_or_err = MemoryBuffer::getFileOrSTDIN(fn);
  if (std::error_code error = file_or_err.getError()) {
    errs() << "Error: Could not open input file '" << fn
           << "': " << error.message() << "\n";
    return nullptr;
  }

  // Load the MLIR module.
  SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(std::move(*file_or_err), SMLoc());
  modules.emplace_back(mlir::parseSourceFile(source_mgr, ctx));
  if (!modules.back()) {
    errs() << "Error: Found no modules in input file '" << fn << "'\n";
    return nullptr;
  }
  return modules.back().get();
}

static void registerDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::StandardOpsDialect>();
  registry.insert<arith::ArithmeticDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<handshake::HandshakeDialect>();
  registry.insert<firrtl::FIRRTLDialect>();
  registry.insert<LLVM::LLVMDialect>();
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "hlt test wrapper generator\n\n");

  mlir::DialectRegistry registry;
  registerDialects(registry);
  mlir::MLIRContext context(registry);

  auto mod1 = getModule(&context, inputFile1);
  if (!mod1)
    return 1;
  auto mod2 = getModule(&context, inputFile2);
  if (!mod2)
    return 1;

  for (mlir::FuncOp funcOp : mod1.getOps<mlir::FuncOp>()) {
    if (funcOp.isPrivate()) {
      auto mod2FuncOps = mod2.getOps<mlir::FuncOp>();
      auto it = llvm::find_if(mod2FuncOps, [&](mlir::FuncOp f2FuncOp) {
        return f2FuncOp.getName() == funcOp.getName() &&
               f2FuncOp.getType() == funcOp.getType();
      });
      if (it != mod2FuncOps.end()) {
        // Found a match, clone the definition from f2 into f1
        BlockAndValueMapping m;
        FuncOp f = *it;
        f.cloneInto(funcOp, m);
        funcOp.setPublic();
      }
    }
  }

  mod1->print(llvm::outs());
}
