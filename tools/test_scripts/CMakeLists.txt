# ===- CMakeLists.txt - ---------------------------------------*- cmake -*-===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===//
#
# This directory contains various test scripts that are generated through CMake
# configuration.
#
# ===-----------------------------------------------------------------------===//

set(SOURCES
  dyn_incrementally_lower
  dyn_hlt_lower
  dyn_hlt_run_sim
  dyn_hlt_build_sim
)
foreach(file IN ITEMS ${SOURCES})
  configure_file(${file}.in ${CIRCT_HLS_TOOLS_DIR}/${file} @ONLY)
  list(APPEND OUTPUTS ${CIRCT_HLS_TOOLS_DIR}/${file})
endforeach()
add_custom_target(circt-hls-testscripts SOURCES ${OUTPUTS})
