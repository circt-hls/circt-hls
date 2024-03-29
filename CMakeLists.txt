cmake_minimum_required(VERSION 3.13.4)

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
  project(circt-hls LANGUAGES CXX C)
  find_package(CIRCT REQUIRED CONFIG)

  option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
  option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)

  if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHs-c- /GR-")
  else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -fno-rtti")
  endif ()

  message(STATUS "Using CIRCTConfig.cmake in: ${CIRCT_DIR}")
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(CIRCT_HLS_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

  list(APPEND CMAKE_MODULE_PATH "${CIRCT_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)

  # todo(fix me): Temporary fix for ensuring that CIRCT_BINARY_DIR is defined.
  if(NOT DEFINED CIRCT_BINARY_DIR)
    # As a fallback, we assume that CIRCTConfig is located somewhere like
    # circt/build/lib/cmake/circt
    # ... as long as it fixes CI, it's good enough for now.
    set(CIRCT_BINARY_DIR "${CIRCT_DIR}/../../..")
  endif()

  include(HandleLLVMOptions)
else ()
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir ) # --src-root
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include ) # --includedir
  set(MLIR_TABLEGEN_OUTPUT_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_TABLEGEN_EXE $<TARGET_FILE:mlir-tblgen>)
  set(CIRCT_BINARY_DIR ${CMAKE_BINARY_DIR})
  set(CIRCT_HLS_BINARY_DIR ${CMAKE_BINARY_DIR})
  include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
  include_directories(SYSTEM ${MLIR_TABLEGEN_OUTPUT_DIR})
  include_directories(SYSTEM ${CIRCT_MAIN_INCLUDE_DIR})
  include_directories(SYSTEM ${CIRCT_INCLUDE_DIR})
endif ()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CIRCT_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})

add_definitions(${LLVM_DEFINITIONS})

#-------------------------------------------------------------------------------
# CIRCT HLS configuration
#-------------------------------------------------------------------------------

# CIRCT HLS project.
set(CIRCT_HLS_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR} ) # --src-root
set(CIRCT_HLS_MAIN_INCLUDE_DIR ${CIRCT_HLS_MAIN_SRC_DIR}/include)

set(CIRCT_HLS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CIRCT_HLS_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include )
set(CIRCT_HLS_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)
set(CIRCT_HLS_UTILS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/utils)
set(CIRCT_HLS_PYTHON_PACKAGES_DIR ${CIRCT_HLS_BINARY_DIR}/python_packages)

list(APPEND CMAKE_MODULE_PATH "${MLIR_MAIN_SRC_DIR}/cmake/modules")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

# Add CIRCT HLS files to the include path
include_directories(${CIRCT_HLS_MAIN_INCLUDE_DIR})
include_directories(${CIRCT_HLS_INCLUDE_DIR})

#-------------------------------------------------------------------------------
# Polygeist Configuration
#-------------------------------------------------------------------------------

# Search for polygeist in path
find_program(MLIR_CLANG_PATH "mlir-clang"
  PATHS
  "${CMAKE_CURRENT_SOURCE_DIR}/Polygeist/build/bin"
  "${CMAKE_BINARY_DIR}/bin"
  NO_DEFAULT_PATH
)
if(NOT EXISTS ${MLIR_CLANG_PATH})
  message(FATAL_ERROR "Did not find mlir-clang in PATH nor in tree; 
  make sure you have built Polygeist before building circt-hls.")
endif()
message(STATUS "Found mlir-clang")

get_filename_component(POLYGEIST_BINARY_DIR ${MLIR_CLANG_PATH} DIRECTORY)
message(STATUS "Using Polygeist binary directory: ${POLYGEIST_BINARY_DIR}")

#-------------------------------------------------------------------------------
# Verilator Configuration
#-------------------------------------------------------------------------------

# Detect if Verilator is present.
if (NOT DEFINED VERILATOR_PATH)
  find_program(VERILATOR_PATH "verilator" PATHS
    "${CMAKE_CURRENT_SOURCE_DIR}/ext/bin" NO_DEFAULT_PATH)
  find_program(VERILATOR_PATH "verilator")
endif()

if(EXISTS ${VERILATOR_PATH})
  message(STATUS "Found Verilator at ${VERILATOR_PATH}.")

  # Find Verilator version.
  execute_process(COMMAND ${VERILATOR_PATH} --version
    OUTPUT_VARIABLE VERILATOR_VERSION)
  string(REGEX MATCH "Verilator (([0-9]+)\.([0-9]+)) \.*"
    MATCH ${VERILATOR_VERSION})
  # It's gotta be at least v4.200.
  if (${CMAKE_MATCH_1} LESS 4.200)
    message(FATAL_ERROR "CIRCT HLS only supports Verilator version 4.200 and up. \
                          Found version: ${CMAKE_MATCH_1}. You can disable \
                          the Verilator tests with '-DVERILATOR_DISABLE=ON'.")
    set(VERILATOR_PATH "")
  endif()
else()
  set(VERILATOR_PATH "")
  message(FATAL_ERROR "Did not find Verilator in path.")
endif()

#-------------------------------------------------------------------------------
# FUD/Calyx Configuration
#-------------------------------------------------------------------------------

# Search for fud in path
# find_program(FUD_PATH "fud")
# if(NOT EXISTS ${FUD_PATH})
#     message(FATAL_ERROR "Did not find fud in PATH. Make sure you have setup
#     Calyx before building and testing circt-hls.")
# endif() 
# message(STATUS "Found fud at ${FUD_PATH}")

#-------------------------------------------------------------------------------
# Directory setup
#-------------------------------------------------------------------------------

add_subdirectory(include/circt-hls)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)
add_subdirectory(integration_test)
add_subdirectory(cosim_test)
