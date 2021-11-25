//===- PassDetail.h - Transforms Pass class details -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef TRANSFORMS_PASSDETAIL_H
#define TRANSFORMS_PASSDETAIL_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
class MemrefDialect;

// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace memref {
class MemRefDialect;
} // end namespace memref

namespace scf {
class SCFDialect;
}

namespace circt_hls {
#define GEN_PASS_CLASSES
#include "circt-hls/Transforms/Passes.h.inc"

} // namespace circt_hls
} // end namespace mlir

#endif // TRANSFORMS_PASSDETAIL_H
