add_circt_dialect_library(CIRCTHLSCosimTransforms
  Passes.cpp

  DEPENDS
  MLIRCosimTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTHLSCosim
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  )
