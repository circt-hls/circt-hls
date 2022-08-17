add_mlir_library(HLSTransforms
  AffineScalRep.cpp
  # AsyncifyCalls.cpp
  # MaxSSA.cpp
  # RenameFunc.cpp
  # PushConstants.cpp
  # CleanUnregisteredAttrs.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_HLS_MAIN_INCLUDE_DIR}/mlir/Transforms

  LINK_LIBS PUBLIC
  MLIRIR
  # MLIRMemRef
  # MLIRControlFlow
  MLIRSupport
  MLIRTransformUtils

  DEPENDS
  HLSTransformsPassIncGen
)
