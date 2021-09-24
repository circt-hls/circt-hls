# circt-hls

**What is this?**:
A collection of repositories used to realise various end-to-end high-level synthesis (HLS) flows centering around the CIRCT project.

The `fud` driver within Calyx is used as a general driver for the entire flow.

- [circt-hls](#circt-hls)
  - [HLS flows](#hls-flows)
- [Setup](#setup)
  - [Build CIRCT](#build-circt)
  - [Build Polygeist](#build-polygeist)
  - [Build Calyx](#build-calyx)
    - [building the rust compiler:](#building-the-rust-compiler)
    - [building the Calyx driver:](#building-the-calyx-driver)
  - [Setting up the HLS `fud` stages](#setting-up-the-hls-fud-stages)
- [Usage](#usage)
  - [Statically scheduled](#statically-scheduled)
  - [Dynamically scheduled](#dynamically-scheduled)
  - [Vivado](#vivado)

## HLS flows

These are the (intended) end-to-end flows that can be run from this directory:
- C
  - Polygeist
    - CIRCT
      - Staticlogic (Statically scheduled HLS)
        - Calyx
          - *Calyx native compiler* (**not circt**)
            - **Verilog**
      - Handshake (dynamically scheduled HLS)
        - FIRRTL
          - HW
            - **Verilog**
  - Vivado ("Classical" HLS)
    - **Verilog**

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

## Build Calyx
**Note: skip this step if you already have a Calyx build available.**

### building the rust compiler:
```
cd calyx
cargo build
```

### building the Calyx driver:
Install Flit:
```
pip3 install flit
```
Install fud:

```
cd calyx
flit -f fud/pyproject.toml install -s
```

Configure fud:
```
cd calyx
fud config global.futil_directory $(pwd)/target/debug/futil
```

Check the fud configuration:
```
fud check
```

If anything relevant is missing, make sure you have the corresponding application installed and available in your path.

## Setting up the HLS `fud` stages
Setup Polygeist/MLIR/CIRCT stages:  
**NOTE: modify these paths to point to your own CIRCT/MLIR builds**.
```
fud register polygeist -p $(pwd)/stages/Polygeist/stage.py
fud config external-stages.polygeist.exec $(pwd)/Polygeist/build/mlir-clang/mlir-clang

fud register circt -p $(pwd)/stages/CIRCT/stage.py
fud config external-stages.circt.bin_dir $(pwd)/../circt/build/bin

fud register mlir -p $(pwd)/stages/MLIR/stage.py
fud config external-stages.mlir.bin_dir $(pwd)/../circt/llvm/build/bin
```

We use the setting
> `fud config stages.circt_hls.toplevel ${toplevel}`

to keep track of the top-level function to be compiled. This should match a function name in the input `.c` file. This will further be used as the top-level function when lowering `SCFToCalyx`.

To initialize this, please run:
```
fud config stages.circt_hls.toplevel ""
```

finally, run `fud config` to ensure that the variables were written correctly.

# Usage
The following lists example commands for exercising the available flows. If you're interested about the specific passes that are getting executed through `fud`, add `--verbose` to the command line.

## Statically scheduled
- [x] Polygeist to **mlir (scf)**
```
fud exec "Polygeist/mlir-clang/Test/aff.c"  \
  --from c                                  \
  --to mlir-scf                             \
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