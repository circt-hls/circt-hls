set(COSIM_SRCS
  CosimOps.cpp
  CosimDialect.cpp
  )

add_circt_dialect_library(CIRCTHLSCosim
  ${COSIM_SRCS}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include

  LINK_LIBS PUBLIC
  MLIRStandard
  MLIRSCF
  MLIRIR

  DEPENDS
  MLIRCosimEnumsIncGen
  MLIRCosimAttrsIncGen
  MLIRCosimTransformsIncGen
  )

add_subdirectory(Transforms)
