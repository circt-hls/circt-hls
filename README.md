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
Checkout https://github.com/llvm/circt and go through the instructions.

## Build Polygeist
We'll build Polygeist using our existing MLIR/LLVM/Clang build. **modify these paths to point to where you built the project**.
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
make sure that the follwing things were available:
- global
- stages.futil.exec
- stages.verilog.exec
- stages.synth-verilog.exec
- stages.vivado-hls.exec

If any of these are missing, make sure you have the corresponding application installed and available in your path.

## Setting up the HLS `fud` stages
Setup polygeist stage:
```
fud register polygeist -p $(pwd)/stages/Polygeist/stage.py
fud config external-stages.polygeist.exec $(pwd)/Polygeist/build/mlir-clang/mlir-clang
```

setup CIRCT/MLIR stage: **modify to point to your own circt build**
```
fud register circt -p $(pwd)/stages/CIRCT/stage.py
fud config external-stages.circt.bin_dir $(pwd)/../circt/build/bin
fud config external-stages.circt.llvm_bin_dir $(pwd)/../circt/llvm/build/bin
```

We use the setting
> `fud config stages.circt.toplevel ${toplevel}`

to keep track of the top-level function to be compiled. This should match a function name in the input `.c` file. This will further be used as the top-level function when lowering `SCFToCalyx`.

To initialize this, please run:
```
fud config stages.circt.toplevel ""
```

# Usage

## Statically scheduled
Polygeist to **mlir (scf)**
```
fud exec "Polygeist/mlir-clang/Test/aff.c"  \
  --from c                                  \
  --to mlir-scf                             \
  -s circt.toplevel "kernel_deriche"`
```

Polygeist to **mlir (calyx)**
```
fud exec "Polygeist/mlir-clang/Test/aff.c"  \
  --from c                                  \
  --to mlir-calyx                           \
  -s circt.toplevel "kernel_deriche"`
```

Polygeist to **calyx**
```
fud exec "Polygeist/mlir-clang/Test/aff.c"  \
  --from c                                  \
  --to calyx                                \
  -s circt.toplevel "kernel_deriche"`
```

## Dynamically scheduled
...
## Vivado
...