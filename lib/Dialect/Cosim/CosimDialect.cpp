//===- CosimDialect.cpp - Implement the Cosim dialect -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Cosim dialect.
//
//===----------------------------------------------------------------------===//

#include "circt-hls/Dialect/Cosim/CosimDialect.h"
#include "circt-hls/Dialect/Cosim/CosimOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt_hls;
using namespace circt_hls::cosim;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void CosimDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt-hls/Dialect/Cosim/Cosim.cpp.inc"
      >();
}

// Provide implementations for the enums, attributes and interfaces that we use.
#include "circt-hls/Dialect/Cosim/CosimAttrs.cpp.inc"
#include "circt-hls/Dialect/Cosim/CosimDialect.cpp.inc"
#include "circt-hls/Dialect/Cosim/CosimEnums.cpp.inc"
