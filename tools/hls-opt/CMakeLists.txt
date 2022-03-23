set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_tool(hls-opt
 hls-opt.cpp
)
llvm_update_compile_flags(hls-opt)
target_link_libraries(hls-opt
  PRIVATE

  HLSTransforms
  HLSCosim
  HLSCosimTransforms

  CIRCTAffineToStaticLogic
  CIRCTAnalysisTestPasses
  CIRCTCalyx
  CIRCTCalyxToHW
  CIRCTCalyxTransforms
  CIRCTESI
  CIRCTExportVerilog
  CIRCTFIRRTL
  CIRCTFIRRTLToHW
  CIRCTFIRRTLTransforms
  CIRCTFSM
  CIRCTHandshake
  CIRCTHandshakeToFIRRTL
  CIRCTHandshakeTransforms
  CIRCTLLHD
  CIRCTLLHDToLLVM
  CIRCTLLHDTransforms
  CIRCTMSFT
  CIRCTHW
  CIRCTHWToLLHD
  CIRCTSCFToCalyx
  CIRCTScheduling
  CIRCTSchedulingTestPasses
  CIRCTSeq
  CIRCTStandardToHandshake
  CIRCTStandardToStaticLogic
  CIRCTStaticLogicOps
  CIRCTSV
  CIRCTSVTransforms

  
  MLIRIR
  MLIRLLVMIR
  MLIRMemRef
  MLIROptLib
  MLIRParser
  MLIRControlFlow
  MLIRSupport
  MLIRTransforms
  MLIRSCF
  )
