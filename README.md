# circt-hls

**What is this?**
A collection of repositories and tools used to realise various end-to-end high-level synthesis (HLS) flows centering around the CIRCT project. This project can be seen as an incubator for things to go into CIRCT (... which itself is an incubator project). If you're researching CIRCT/MLIR based HLS flows, feel free to contribute or push your code to this repository - there are as of now no strict review or code requirements.

**What tools do i get?**
- **Polygeist:** A clang-based C-to-MLIR compiler front-end.
- **CIRCT tools:** MLIR-based hardware compiler(s) and tools
- **Calyx native compiler:** A rust compiler for the [Calyx](https://calyxir.org/) Hardware IR
- **HSDbg:** A visual debugging tool for Handshake (dataflow) circuits
- **HLT:** An MLIR-to-Verilator cosimulation framework
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

## Build CIRCT
If you don't already have a CIRCT/MLIR build locally, checkout https://github.com/llvm/circt and go follow the instructions.

## Build Polygeist
Polygeist is used as the primary front-end for ingesting high-level programs. We'll build Polygeist using our existing MLIR/LLVM/Clang build.  
**NOTE: modify these paths to point to your own CIRCT/MLIR builds**.
```
cd Polygeist
mkdir build
cd build
cmake -G Ninja .. \
  -DMLIR_DIR=$PWD/../circt/llvm-project/build/lib/cmake/mlir \
  -DCLANG_DIR=$PWD/../circt/llvm-project/build/lib/cmake/clang \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
ninja
ninja check-mlir-clang
```

## Setup and build Calyx and the HLS `fud` stages

The `fud` driver within Calyx is used as a general driver for the entire flow.
Run the `calyx_setup.sh` script from the root repository folder. If any step fails due to missing dependencies, download the dependencies and rerun the script.

We use the setting
> `fud config stages.circt_hls.toplevel ${toplevel}`

to keep track of the top-level function to be compiled. This should match a function name in the input `.c` file. This will further be used as the top-level function when lowering `SCFToCalyx`.

To initialize this, please run:
```
fud config stages.circt_hls.toplevel ""
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
```
(Modify the above wrt. where you built circt/mlir/llvm).

# Usage
The following lists example commands for exercising the available flows. If you're interested about the specific passes that are getting executed through `fud`, add `--verbose` to the command line.

## Statically scheduled

### Pipeline:

- Polygeist
- MLIR SCF for-to-while
  - SCFtoCalyx only supports while loops
- MLIR SCF to Calyx
- Export Calyx IR
- Calyx Native compilation

### Status
<details>
  <summary>Click to expand</summary>
- [x] Polygeist to **mlir (scf)**
```
fud exec "Polygeist/mlir-clang/Test/aff.c"  \
  --from c                                  \
  --to mlir-scf-while                       \
  -s circt_hls.toplevel "kernel_deriche"
```

- [ ] Polygeist to **mlir (calyx)**  
**Error:** need to lower for-to-while loops (still waiting for https://reviews.llvm.org/D108454).
```
fud exec "Polygeist/mlir-clang/Test/aff.c"  \
  --from c                                  \
  --to mlir-calyx                           \
  -s circt_hls.toplevel "kernel_deriche"
```

- [ ] Polygeist to **calyx**  
**Error:** invalid lowering of comb groups
```
fud exec "Polygeist/mlir-clang/Test/aff.c"  \
  --from c                                  \
  --to futil                                \
  -s circt_hls.toplevel "kernel_deriche"
```
</details>

## Dynamically scheduled

### Compilation pipeline

- Polygeist
- MLIR SCF to std
- CIRCT flatten memrefs
  - Ensures multidimensional memories are flatten to unidimensional memories (what is currently supported in Handshake-to-FIRRTL lowering)
- CIRCT std to handshake + canonicalization
- CIRCT handshake bufferization
  - Currently, the `all` mode is used since `cycles` introduces deadlocks in the majority of circuits.
- CIRCT Handshake to FIRRTL
- CIRCT Firtool FIRRTL to sv
  - We trust firtool to do the right things in the right order to bring our FIRRTL code all the way to hardware.
### Status

<details>
  <summary>Click to expand</summary>

- [ ] Polygeist to **mlir (handshake)**  
**Error:** issues in lowering the memref.alloc ops in standard to handshake. I think this goes back to what https://github.com/llvm/circt/pull/1538 is trying to solve.
```
fud exec "examples/c/fir/fir.c" \
  --from c                      \
  --to mlir-handshake           \
  -s circt_hls.toplevel fir
```

- [ ] Polygeist to **mlir (FIRRTL)**  
**error:** issues in FIRRTL with unbounded memories
```
fud exec "examples/c/fir/fir.c" \
  --from c                      \
  --to mlir-firrtl              \
  -s circt_hls.toplevel fir
```

- [X] Handshake to Verilog
```
fud exec ${handshake MLIR file} \
  --from mlir-handshake         \
  --to synth-verilog

- [X] Handshake to synthesized
```
fud exec ${handshake MLIR file} \
  --from mlir-handshake         \
  --to synth-files -o ${outdir}

</details>

## Vivado
...
