set(COSIM_SRCS
  CosimOps.cpp
  CosimDialect.cpp
  )

add_mlir_dialect_library(HLSCosim
  ${COSIM_SRCS}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include

  LINK_LIBS PUBLIC
  MLIRControlFlow
  MLIRSCF
  MLIRIR

  DEPENDS
  MLIRCosimEnumsIncGen
  MLIRCosimAttrsIncGen
  MLIRCosimTransformsIncGen
  )

add_subdirectory(Transforms)
