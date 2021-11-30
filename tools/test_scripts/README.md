# CIRCT-HLS test scripts

This directory contains the main scripts used for testing the end-to-end HLS flows
defined through this repository. Each of the scripts are defined as `.in` files.
These will be generated on running `cmake`, wherein build-specific paths will be 
supstituted, and the scripts will be placed in the `circt-hls/build/bin` directory where you can reference them directly, if need be.
All of the scripts will generate intermediate output in the working directory where
the script was called from.
When running `ninja check-circt-hls-integration-cosim` this means that all of the
intermediate output can be located in the `build/cosim_test/...` directories.

In general, all of these scripts expects the following:
- A kernel function has the same name as its filename
- There exist a testbench for the kernel in the same directory of the kernel.
- The testbench is named `tst_${kernelname}.c`
- The testbench has a `main` function.
- Scripts take the path of the `tst_${kernelname}.c` as input.

The scripts are as follows:
- **dyn_hlt_incrementally_lower**:
  - A script which uses `fud` to lower a kernel from C to system verilog.
- **dyn_hlt_lower**:
  - A script which references `dyn_hlt_incrementally_lower` to first lower a kernel, and then builds an HLT wrapper around it (for cosimulation).
- **dyn_hlt_build_sim**:
  - A script which builds a HLT simulation library; this is done through Verilating the lowered kernel and compiling this together with a HLT wrapper.
- **dyn_hlt_build_tb**:
  - A script which lowers a C testbench into an HLT-compatible testbench.
- **dyn_hlt_run_sim**:
  - A script which executes a testbench through `mlir-cpu-runner` together with a simulator library. The testbench is expected to be lowered to LLVM MLIR (built through `dyn_hlt_build_tb`) and the testbench library to be built through `dyn_hlt_build_sim`.
- **hlt_test**:
  - A script that generates driver scripts for a testbench, and defines the overall test flow. These driver scripts are convenient if you want to reproduce certain steps of the synthesis/simulation pipeline. If so, just navigate to the test output directory and run the script in question.
- **run_test_scripts**:
  - A script which will execute all scripts that are piped to it via. stdin. Used together with `hlt_test`.
