add_mlir_dialect(Cosim cosim)
add_mlir_doc(Cosim -gen-dialect-doc Cosim Dialects/)

set(LLVM_TARGET_DEFINITIONS Cosim.td)

mlir_tablegen(CosimAttrs.h.inc -gen-struct-attr-decls)
mlir_tablegen(CosimAttrs.cpp.inc -gen-struct-attr-defs)
add_public_tablegen_target(MLIRCosimAttrsIncGen)
add_dependencies(mlir-headers MLIRCosimAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS CosimPasses.td)
mlir_tablegen(CosimPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRCosimTransformsIncGen)
add_mlir_doc(CosimPasses -gen-pass-doc CosimPasses ./)
