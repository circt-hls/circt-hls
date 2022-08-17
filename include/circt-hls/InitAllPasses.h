//===- InitAllPasses.h - CIRCT Global Pass Registration ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all passes to the
// system.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_HLS_INITALLPASSES_H_
#define CIRCT_HLS_INITALLPASSES_H_

// #include "circt-hls/Dialect/Cosim/CosimPasses.h"
#include "circt-hls/Transforms/Passes.h"

namespace circt_hls {

inline void registerAllPasses() {
  // Transformation passes
  registerTransformsPasses();

  // Dialect Passes
  // cosim::registerPasses();
}

} // namespace circt_hls

#endif // CIRCT_HLS_INITALLPASSES_H_
