# circt-hls

**What is this?**:
A collection of repositories used to realise various end-to-end high-level synthesis (HLS) flows centering around the CIRCT project.

The `fud` driver within Calyx is used as a general driver for the entire flow.

- [circt-hls](#circt-hls)
  - [HLS flows](#hls-flows)
- [Setup](#setup)
  - [Build CIRCT](#build-circt)
  - [Build Polygeist](#build-polygeist)
  - [Setup Calyx and the HLS fud stages](#setup-calyx-and-the-hls-fud-stages)
- [Usage](#usage)
  - [Statically scheduled](#statically-scheduled)
  - [Dynamically scheduled](#dynamically-scheduled)
  - [Vivado](#vivado)

## HLS flows

These are the (intended) end-to-end flows that can be run from this directory:
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
  - [Vivado](https://en.wikipedia.org/wiki/Xilinx_Vivado) ("Classical" HLS)
    -  [**SystemVerilog**](https://en.wikipedia.org/wiki/SystemVerilog)

# Setup

## Build CIRCT
If you don't already have a CIRCT/MLIR build locally, checkout https://github.com/llvm/circt and go follow the instructions.

## Build Polygeist
We'll build Polygeist using our existing MLIR/LLVM/Clang build.  
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

## Setup Calyx and the HLS fud stages

run the `calyx_setup.sh` script from the root repository folder. If any step fails due to missing dependencies, download the dependencies and rerun the script.

We use the setting
> `fud config stages.circt_hls.toplevel ${toplevel}`

to keep track of the top-level function to be compiled. This should match a function name in the input `.c` file. This will further be used as the top-level function when lowering `SCFToCalyx`.

To initialize this, please run:
```
fud config stages.circt_hls.toplevel ""
```

# Usage
The following lists example commands for exercising the available flows. If you're interested about the specific passes that are getting executed through `fud`, add `--verbose` to the command line.

## Statically scheduled
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

## Dynamically scheduled

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


## Vivado
...
