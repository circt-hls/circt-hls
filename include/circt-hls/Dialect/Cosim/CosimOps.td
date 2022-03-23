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
    StrAttr:$ref,
    StrArrayAttr:$targets,
    Variadic<AnyType>:$operands);

  let results = (outs Variadic<AnyType>);
  let assemblyFormat = [{
    $func `(` $operands `)` `:` functional-type($operands, results) attr-dict
  }];

  let hasVerifier = 1;
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

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<
    (ins "mlir::SymbolRefAttr":$func, "mlir::TypeRange":$results, "mlir::StringRef":$ref,
      "mlir::ArrayRef<std::string>":$targets, CArg<"mlir::ValueRange", "{}">:$operands)>
  ];
}

def CompareOp : Cosim_Op<"compare", [
    SameTypeConstraint<"ref", "target">
  ]> {
  let summary = "Compares a reference value with a cosimulated value";
  let description = [{
    Wrap calls that should be cosimulated and coverified

    Example:
    ```mlir
    cosim.compare %1, %2 : memref<100xi32>
    ```
  }];

  let arguments = (ins
    AnyType:$ref, // Reference value to compare
    AnyType:$target, // target value to compare
    OptionalAttr<FlatSymbolRefAttr>:$ref_src,
    OptionalAttr<FlatSymbolRefAttr>:$target_src
  );
  let assemblyFormat = "$ref `,` $target `:` type($ref) attr-dict";
  let extraClassDeclaration = [{
    // Get the type of the operands that are compared
    mlir::Type type() {
      return ref().getType();
    }
  }];
}
