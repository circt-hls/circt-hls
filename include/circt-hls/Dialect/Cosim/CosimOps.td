//===- Ops.td - Cosim operation definitions ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//============================================================================//
//
// This file define Cosim ops in tablegen.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/CallInterfaces.td"

// InstanceOp
def CallOp : Cosim_Op<"call", [
    CallOpInterface
  ]> {
  let summary = "Wrap calls that should be cosimulated and coverified";
  let description = [{
    Wrap calls that should be cosimulated and coverified

    Example:
    ```mlir
    %2 = memref.subview %1[0, 0] [1, 100] [1, 1] : memref<1x100xi32> to memref<100xi32>
    cosim.wrap {
      call @simple_example_1(%2) : (memref<100xi32>) -> ()
    } {
      targets = ["simple_example_1", "simple_example_1_hs"],
      ref = "simple_example_1"
    }
    ```
  }];

  let arguments = (ins 
    FlatSymbolRefAttr:$func,
    AnyAttr:$ref,
    ArrayAttr:$targets,
    Variadic<AnyType>:$operands);

  let results = (outs Variadic<AnyType>);
  let assemblyFormat = [{
    $func `(` $operands `)` `:` functional-type($operands, results) attr-dict
  }];

  let verifier = "return ::verify$cppClass(*this);";
  let extraClassDeclaration = [{
    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }


    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("func");
    }
  }];
}

def CompareOp : Cosim_Op<"compare", [
    SameTypeConstraint<"ref", "cosim">
  ]> {
  let summary = "Compares a reference value with a cosimulated value";
  let description = [{
    Wrap calls that should be cosimulated and coverified

    Example:
    ```mlir
    cosim.compare(%1 : memref<100xi32>, %2 : memref<100xi32>)
    ```
  }];

  let arguments = (ins AnyType:$ref, AnyType:$cosim);
  let assemblyFormat = "$ref `,` $cosim `:` type($ref) attr-dict";
}
