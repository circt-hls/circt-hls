add_circt_dialect(Cosim cosim)
add_circt_doc(Cosim -gen-dialect-doc Cosim Dialects/)

set(LLVM_TARGET_DEFINITIONS Cosim.td)

mlir_tablegen(CosimEnums.h.inc -gen-enum-decls)
mlir_tablegen(CosimEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRCosimEnumsIncGen)
add_dependencies(circt-headers MLIRCosimEnumsIncGen)

mlir_tablegen(CosimAttrs.h.inc -gen-struct-attr-decls)
mlir_tablegen(CosimAttrs.cpp.inc -gen-struct-attr-defs)
add_public_tablegen_target(MLIRCosimAttrsIncGen)
add_dependencies(circt-headers MLIRCosimAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS CosimPasses.td)
mlir_tablegen(CosimPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRCosimTransformsIncGen)
add_circt_doc(CosimPasses -gen-pass-doc CosimPasses ./)
