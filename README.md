# circt-hls


# Setup

## Build Polygeist
You are most likely a CIRCT developer, so it is recommended to point Polygeist to an existing MLIR/LLVM/Clang build:
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