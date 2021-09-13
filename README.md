# circt-hls

### What is this?
A collection of repositories used to realise various end-to-end high-level synthesis (HLS) flows centering around the CIRCT project.

The `fud` driver within Calyx is used as a general driver for the entire flow.

## HLS flows

These are the (intended) end-to-end flows that can be run from this directory:
- C
  - Polygeist
    - CIRCT
      - Staticlogic (Statically scheduled HLS)
        - Calyx
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
We'll build Polygeist using our existing MLIR/LLVM/Clang build:
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

## Usage
