//===- hls-opt.cpp - The hls-opt driver -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'hls-opt' tool, which is the hls analog of
// mlir-opt, used to drive compiler passes, e.g. for testing.
//
//===----------------------------------------------------------------------===//

#include "circt-hls/InitAllDialects.h"
#include "circt/Support/LoweringOptions.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "circt-hls/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register MLIR stuff
  registry.insert<mlir::AffineDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::StandardOpsDialect>();
  registry.insert<mlir::arith::ArithmeticDialect>();
  registry.insert<mlir::scf::SCFDialect>();

  circt_hls::registerAllDialects(registry);

  // Register the standard passes we want.
  mlir::registerCSEPass();
  mlir::registerSCCPPass();
  mlir::registerInlinerPass();
  mlir::registerCanonicalizerPass();
  circt_hls::registerTransformsPasses();

  return mlir::failed(mlir::MlirOptMain(
      argc, argv, "CIRCT HLS modular optimizer driver", registry,
      /*preloadDialectsInContext=*/false));
}
