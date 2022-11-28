# circt-hls

[![Build](https://github.com/circt-hls/circt-hls/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/circt-hls/circt-hls/actions/workflows/build.yml) [![Test](https://github.com/circt-hls/circt-hls/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/circt-hls/circt-hls/actions/workflows/test.yml)

**What is this?**
A collection of repositories and tools used to realise various end-to-end high-level synthesis (HLS) flows centering around the CIRCT project. This project can be seen as an incubator for things to go into CIRCT (... which itself is an incubator project). If you're researching CIRCT/MLIR based HLS flows, feel free to contribute or push your code to this repository - there are as of now no strict review or code requirements.

**What tools do i get?**
- **Polygeist:** A clang-based C-to-MLIR compiler front-end.
- **CIRCT tools:** MLIR-based hardware compiler(s) and tools
- **Calyx native compiler:** A rust compiler for the [Calyx](https://calyxir.org/) Hardware IR
- **HSDbg:** A visual debugging tool for Handshake (dataflow) circuits
- **HLT:** An MLIR-to-Verilator cosimulation framework
- **Cosim:** A dialect to support cosimulation and coverification
- **hls-opt:** Various HLS-related MLIR passes, mostly to support the above tools.

These are the (intended) end-to-end flows that can be driven from this directory:
- [C](https://en.wikipedia.org/wiki/C_(programming_language))
  - [Polygeist](https://c.wsmoses.com/papers/Polygeist_PACT.pdf)
    - [CIRCT](https://circt.llvm.org/)
      - [Staticlogic Dialect](https://circt.llvm.org/docs/Dialects/StaticLogic/) (statically scheduled HLS)
        - [Calyx Dialect](https://circt.llvm.org/docs/Dialects/Calyx/)
          - [Calyx *native compiler*](https://capra.cs.cornell.edu/calyx/) (**not** a component of CIRCT)
            - [**SystemVerilog**](https://en.wikipedia.org/wiki/SystemVerilog)
      - [Handshake Dialect](https://circt.llvm.org/docs/Dialects/Handshake/) (dynamically scheduled HLS)
        - [FIRRTL Dialect](https://circt.llvm.org/docs/Dialects/FIRRTL/)
          - [HW Dialect](https://circt.llvm.org/docs/Dialects/HW/)
            - [**SystemVerilog**](https://en.wikipedia.org/wiki/SystemVerilog)
            - HLS Cosimulation
  - [Vivado](https://en.wikipedia.org/wiki/Xilinx_Vivado) ("Classical" HLS)
    -  [**SystemVerilog**](https://en.wikipedia.org/wiki/SystemVerilog)

# Setup

**Since things are changing rapidly, the most reliable method of setting this up and getting things to work is to [replicate the steps done in CI](https://github.com/circt-hls/circt-hls/blob/main/.github/workflows/actions/build_circt_hls/action.yml) (minus the caching steps).**

## Build CIRCT
If you don't already have a CIRCT/MLIR build locally, checkout https://github.com/llvm/circt and go follow the instructions. We **do not** include a CIRCT submodule in `circt-hls` since we assume that most people interested in this project will already be looking into CIRCT.

CIRCT-HLS was successfully compiled (and mostly successfully tested) with CIRCT version [062eab31a67c8a8418b2ba9fb4f1814cdca08783](https://github.com/llvm/circt/commit/062eab31a67c8a8418b2ba9fb4f1814cdca08783).  The test failure case: cosim :: suites/Dynamatic/insertion_sort/tst_insertion_sort.c

The script `build.sh` (tested under Ubuntu 22.04.1) may also provide a reference point for building the tool chain.

## Build Polygeist
Polygeist is used as the primary front-end for ingesting high-level programs. While it *is* possible to build Polygeist with an external LLVM/MLIR build, it often occurs that the CIRCT and Polygeist LLVM versions have API-breaking differences. It is suggested that you try building Polygeist with an external LLVM build (reusing your CIRCT build), and as a fallback use the internal LLVM submodule of Polygeist, which is sure to work.
Information on how to build Polygeist is available in the [Polygeist repository](https://github.com/wsmoses/Polygeist).

## Install Python dependencies

Some tooling relies on certain Python packages, which can be installed using `requirements.txt`:

```
pip install -r requirements.txt
```

## Build and test CIRCT-HLS 

This repository contains an LLVM tool called `circt-hls` which provides various passes used to realize the HLS flows. This is the primary place for incubating things which are intended to be eventually merged into CIRCT. 

To build the tool:
```
$ mkdir build
$ cd build
$ cmake -G Ninja .. \
    -DCIRCT_DIR=$PWD/../circt/build/lib/cmake/circt \
    -DMLIR_DIR=$PWD/../circt/llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../circt/llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
$ ninja
$ ninja check-circt-hls
$ ninja check-circt-hls-integration
$ ninja check-circt-hls-cosim
```
(Modify the above wrt. where you built circt/mlir/llvm).

# Usage

CIRCT-HLS is driven using the `hlstool`, available in the `build/bin` path where you built CIRCT-HLS. A guide to using `hlstool` is available [here](tools/hlstool/README.md).

A good starting point for getting familiar with the flow is to execute the cosimulation test suite (see below). These tests will emit all of the IR that was generated during lowering, which can be manually inspected. From here, we encourage you to write your own kernels! For inspiration on how to drive the tool, the best reference is to replicate the commands in the [cosimulation test suite](https://github.com/circt-hls/circt-hls/tree/main/cosim_test/suites/Dynamatic) (the `// RUN:` line can be executed manually, substituting the `%s` argument with the path of the source file).

## Tests:
- integration tests can be run by executing the `ninja check-circt-hls-integration` command in the `circt-hls/build` directory. This will execute the `lit` integration test suites.
- Cosimulation verification can be run by executing the `ninja check-circt-hls-cosim` command in the `circt-hls/build` directory. This will execute the `lit` extended integration test suite, HLS'ing all of the C tests in the `cosim_test` directory. Each file is progressively lowered and the intermediate representations for each file during the lowering process will be available in `build/cosim_test/suites/Dynamatic/...`. This can be very helpful if you're developing and want to inspect (or use) some of the intermediate results generated during compilation.
