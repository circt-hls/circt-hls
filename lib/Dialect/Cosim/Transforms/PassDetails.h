//===- PassDetails.h - Cosim pass class details -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the stuff shared between the different Cosim passes.
//
//===----------------------------------------------------------------------===//

// clang-tidy seems to expect the absolute path in the
// header guard on some systems, so just disable it.
// NOLINTNEXTLINE(llvm-header-guard)
#ifndef DIALECT_COSIM_TRANSFORMS_PASSDETAILS_H
#define DIALECT_COSIM_TRANSFORMS_PASSDETAILS_H

#include "circt-hls/Dialect/Cosim/CosimOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace scf {
class SCFDialect;
}
namespace LLVM {
class LLVMDialect;
}

} // namespace mlir

namespace circt_hls {
namespace cosim {

#define GEN_PASS_CLASSES
#include "circt-hls/Dialect/Cosim/CosimPasses.h.inc"

} // namespace cosim
} // namespace circt_hls

#endif // DIALECT_COSIM_TRANSFORMS_PASSDETAILS_H
