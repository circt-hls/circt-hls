//===- CosimPasses.h - Cosim pass entry points ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_COSIM_COSIMPASSES_H
#define CIRCT_DIALECT_COSIM_COSIMPASSES_H

#include "circt/Support/LLVM.h"
#include <map>
#include <memory>
#include <set>


namespace circt_hls {
namespace cosim {
class FuncOp;

std::unique_ptr<mlir::Pass> createCosimLowerCallPass();
std::unique_ptr<mlir::Pass> createCosimLowerComparePass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt-hls/Dialect/Cosim/CosimPasses.h.inc"

} // namespace cosim
} // namespace circt_hls

#endif // CIRCT_DIALECT_COSIM_COSIMPASSES_H
