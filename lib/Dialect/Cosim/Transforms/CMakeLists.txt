add_mlir_dialect_library(HLSCosimTransforms
  Passes.cpp

  DEPENDS
  MLIRCosimTransformsIncGen

  LINK_LIBS PUBLIC
  HLSCosim
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  )
