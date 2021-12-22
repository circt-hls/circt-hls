get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(circt_dialect_libs GLOBAL PROPERTY CIRCT_DIALECT_LIBS)

add_llvm_executable(mlir-resolve mlir-resolve.cpp mlir-resolve.cpp)

llvm_update_compile_flags(mlir-resolve)
target_link_libraries(mlir-resolve PRIVATE
    ${mlir_dialect_libs}
    ${circt_dialect_libs}
  )
