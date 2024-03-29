﻿//===- VerilatorEmitterUtils.cpp - Verilator emission utilities -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains functions for emitting verilator-compatible types from
// MLIR types.
//
//===----------------------------------------------------------------------===//

#include "circt-hls/Tools/hlt/WrapGen/VerilatorEmitterUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace circt_hls {

LogicalResult emitVerilatorTypeFromWidth(llvm::raw_ostream &os, Location loc,
                                         unsigned width) {
  if (width <= 8)
    os << "CData";
  else if (width <= 16)
    os << "SData";
  else if (width <= 32)
    os << "IData";
  else if (width <= 64)
    os << "QData";
  else
    return emitError(loc)
           << "Integers wider than 64 bits are unhandled for now";
  return success();
}

LogicalResult emitVerilatorType(llvm::raw_ostream &os, Location loc, Type type,
                                Optional<StringRef>) {
  return llvm::TypeSwitch<Type, LogicalResult>(type)
      .Case<IntegerType>([&](IntegerType type) -> LogicalResult {
        unsigned bits = type.getIntOrFloatBitWidth();
        return emitVerilatorTypeFromWidth(os, loc, bits);
      })
      .Case<IndexType>([&](IndexType type) {
        return emitVerilatorType(
            os, loc,
            IntegerType::get(type.getContext(), type.kInternalStorageBitWidth));
      })
      .Case<MemRefType>([&](MemRefType type) {
        if (emitVerilatorType(os, loc, type.getElementType()).failed())
          return failure();
        os << "*";
        return success();
      })
      .Default([&](auto type) {
        return emitError(loc) << "no known conversion from '" << type
                              << "' to a verilator type.";
      });
}

} // namespace circt_hls
