add_mlir_library(HLSTransforms
  AsyncifyCalls.cpp
  MaxSSA.cpp
  RenameFunc.cpp
  PushConstants.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_HLS_MAIN_INCLUDE_DIR}/mlir/Transforms

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRef
  MLIRStandard
  MLIRSupport
  MLIRTransformUtils

  DEPENDS
  HLSTransformsPassIncGen
)
