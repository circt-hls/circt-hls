# ===- CMakeLists.txt - ---------------------------------------*- cmake -*-===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===//

set(SOURCES
  hlstool
)
foreach(file IN ITEMS ${SOURCES})
  configure_file(${file}.in ${CIRCT_HLS_TOOLS_DIR}/${file} @ONLY)
  list(APPEND OUTPUTS ${CIRCT_HLS_TOOLS_DIR}/${file})
endforeach()
add_custom_target(hlstool SOURCES ${OUTPUTS})
