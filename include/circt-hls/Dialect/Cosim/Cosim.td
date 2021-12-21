//===- Cosim.td - Cosim dialect definition -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Cosim dialect.
//
//===----------------------------------------------------------------------===//

#ifndef COSIM_TD
#define COSIM_TD

include "mlir/IR/OpBase.td"

def Cosim_Dialect : Dialect {
  let name = "cosim";
  let cppNamespace = "::circt_hls::cosim";
}

class SameTypeConstraint<string lhs, string rhs>
 : TypesMatchWith<"lhs and rhs types should be equivalent",
                   lhs, rhs, [{ $_self }]>;

// Base class for Cosim dialect ops.
class Cosim_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<Cosim_Dialect, mnemonic, traits> {
}


include "circt-hls/Dialect/Cosim/CosimOps.td"

#endif // COSIM_TD
