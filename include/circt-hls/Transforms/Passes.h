//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for CIRCT transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_HLS_TRANSFORMS_PASSES_H
#define CIRCT_HLS_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <limits>

namespace circt_hls {

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> createAsyncifyCallsPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt-hls/Transforms/Passes.h.inc"

} // namespace circt_hls

#endif // CIRCT_HLS_TRANSFORMS_PASSES_H
