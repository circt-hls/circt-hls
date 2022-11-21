# -*- Python -*-

import os
import platform
import re
import shutil
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'CIRCT-HLS cosim'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir', '.c']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.circt_hls_obj_root, 'cosim_test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))
config.substitutions.append(('%shlibdir', config.circt_shlib_dir))

config.substitutions.append(('%llvm_shlibdir', config.llvm_lib_dir))
config.substitutions.append(('%llvm_obj_root', config.llvm_obj_root))
config.substitutions.append(('%circt_src_root', config.circt_src_root))
config.substitutions.append(('%circt_hls_obj_root', config.circt_hls_obj_root))

llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

tool_dirs = [
    config.circt_hls_tools_dir,
    config.circt_tools_dir, config.mlir_tools_dir, config.llvm_tools_dir,
]
tools = [
    'hls-opt', 'hlt-wrapgen', 'dyn_incrementally_lower', 'dyn_hlt_lower',
    'hlstool'
]

# Enable Polygeist if it has been detected.
if config.polygeist_bin_dir != "":
  tool_dirs.append(config.polygeist_bin_dir)
  tools.append('mlir-clang')
  tools.append('polygeist-opt')

# Maximum 500 seconds for each test. This might be too much but some of these
# tests might be very slow depending on the executing machine capabilities and the
# amount of test parallelism used.
lit_config.maxIndividualTestTime = 500

llvm_config.add_tool_substitutions(tools, tool_dirs)
