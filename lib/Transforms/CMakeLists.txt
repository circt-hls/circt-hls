add_circt_library(CIRCTHLSTransforms
  AsyncifyCalls.cpp
  MaxSSA.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_HLS_MAIN_INCLUDE_DIR}/circt/Transforms

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRef
  MLIRStandard
  MLIRSupport
  MLIRTransformUtils

  DEPENDS
  CIRCTHLSTransformsPassIncGen
)
