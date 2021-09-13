# circt-hls


## Setup

### Build Polygeist
You are most likely a CIRCT developer, so it is recommended to point Polygeist to an existing MLIR/LLVM/Clang build:
```
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

### Build Calyx

