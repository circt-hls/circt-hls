#!/bin/bash -ex
exec &> >(tee "build.log")
# guided by .github/workflows
# tested only with Ubuntu 22.04.1 LTS

# color codes end up in build.log for browsing with 'less -R'
export CLICOLOR_FORCE

git submodule init
git submodule update

# create a local virtual environment for all python packages
sudo apt-get install -y python3.10-venv
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
# if torch-mlir desired
#pip install --pre torch-mlir torchvision -f https://llvm.github.io/torch-mlir/package-index/ --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip install -r requirements.txt

BUILD_TYPE="Debug"
#BUILD_TYPE="Release"

# using existence of directory to mean it is already installed
# rm -rf <dir> to checkout and recompile

# Build Verilator
export VERILATOR_ROOT="$(pwd)/verilator"
if [ ! -d "verilator" ] ; then
  echo "build: verilator"
  sudo apt-get install -y git perl python3 make autoconf g++ flex bison
  sudo apt-get install -y libgoogle-perftools-dev numactl perl-doc ccache
  #sudo apt-get install -y libfl2 libfl-dev zlibc zlib1g zlib1g-dev
  sudo apt-get install -y libfl2 libfl-dev zlib1g zlib1g-dev gdb
  git clone --branch stable --depth 1 https://github.com/verilator/verilator 
  (cd verilator; autoconf; ./configure; make -j $(nproc))
  #make install
  (cd verilator; make test)
fi

export PATH="$PATH:$VERILATOR_ROOT/bin"
# not checked in CMakeLists.txt $ENV{VERILATOR_PATH}
#export VERILATOR_PATH="$VERILATOR_ROOT/bin"

if [ ! -d "circt" ] ; then
  echo "clone: circt"
  # Checkout CIRCT
  git clone https://github.com/llvm/circt 
  (cd circt; 
   git reset --hard 2500cf570bbc6c6d45940466f9421e93d3a49132;
   #git reset --hard 958a453ff737f3bf4df609da4cec1844904d266d; 
   #git reset --hard 76e38b19f6738dfb4d3c8dd26db017ffbf432e06;
   git submodule init; git submodule update)
  #(cd circt; 
  # git checkout sifive/1/22/0;
  # git submodule init; git submodule update)

  # Build CIRCT version of LLVM+MLIR
  echo "build: circt llvm"
  mkdir -p circt/llvm/build
  (cd circt/llvm/build;
         # to use -DLLVM_ENABLE_LLD=ON needs clang 
         # 13.0.1-++20211124043118+19b8368225dc-1~exp1~20211124043708.23
         cmake -G Ninja ../llvm \
              -DLLVM_BUILD_EXAMPLES=OFF \
              -DLLVM_ENABLE_BINDINGS=OFF \
              -DBUILD_SHARED_LIBS=ON \
              -DLLVM_OPTIMIZED_TABLEGEN=ON \
              -DLLVM_ENABLE_PROJECTS="mlir" \
              -DLLVM_TARGETS_TO_BUILD="host" \
              -DLLVM_ENABLE_ASSERTIONS=ON \
              -DCMAKE_BUILD_TYPE=$BUILD_TYPE;
          ninja -j$(nproc);
          ninja check-mlir;
  )
fi

# Build CIRCT itself
echo "build: circt"
mkdir -p circt/build
(cd circt/build;
        #export CC=clang;
        #export CXX=clang++;
        cmake -G Ninja .. \
          -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
          -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
          -DLLVM_ENABLE_ASSERTIONS=ON \
          -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
          -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ;
        ninja -j$(nproc);
        ninja check-circt;
        # Failed Tests (1):
        #  CIRCT :: Dialect/FSM/variable/top.mlir
        #ninja check-circt-integration;  
)

# Build Polygeist and its version of LLVM + MLIR. We do this separately from the CIRCT
# build since there are often API-breaking changes between the two versions...
echo "build: polygeist llvm"
(cd Polygeist; git submodule init; git submodule update;
  mkdir -p llvm-project/build;
  cd llvm-project/build; 
       # -DLLVM_ENABLE_LLD=ON \
       # needs clang 13.0.1-++20211124043118+19b8368225dc-1~exp1~20211124043708.23
       # export CC="clang";
       # export CXX="clang++";
       cmake ../llvm -GNinja \
          -DLLVM_ENABLE_PROJECTS="llvm;clang;mlir;openmp" \
          -DBUILD_SHARED_LIBS=ON \
          -DLLVM_OPTIMIZED_TABLEGEN=ON \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_FLAGS="-Wno-c++11-narrowing";
        ninja -j$(nproc);
        ninja check-mlir;
  )

echo "build: polygeist"
# Build Polygeist itself
mkdir -p Polygeist/build
(cd Polygeist/build;
        cmake -G Ninja .. \
          -DMLIR_DIR=$PWD/../llvm-project/build/lib/cmake/mlir \
          -DCLANG_DIR=$PWD/../llvm-project/build/lib/cmake/clang \
          -DLLVM_TARGETS_TO_BUILD="host" \
          -DLLVM_ENABLE_ASSERTIONS=ON \
          -DCMAKE_BUILD_TYPE=$BUILD_TYPE ; 
        ninja -j$(nproc);
        ninja mlir-clang;
        #fails test/polygeist-opt/lower_polygeist_ops.mlir
        # fix: replace func with func.func
        #ninja check-polygeist-opt;
        # not available in 524ccc8
        #ninja check-cgeist;
)

# Build CIRCT-HLS and test
echo "build: circt-hls"
mkdir -p build
# for mlir-cpu-runner
export PATH=$PATH:$(pwd)/circt/build/bin ;
# for hlsdbg console
sudo apt-get install -y xterm
(cd build;
        cmake -G Ninja .. \
          -DCIRCT_DIR=$PWD/../circt/build/lib/cmake/circt \
          -DMLIR_DIR=$PWD/../circt/llvm/build/lib/cmake/mlir \
          -DLLVM_DIR=$PWD/../circt/llvm/build/lib/cmake/llvm \
          -DLLVM_ENABLE_ASSERTIONS=ON \
          -DCMAKE_BUILD_TYPE=$BUILD_TYPE ;
        ninja -j$(nproc) ;
        ninja check-circt-hls ;
        ninja check-circt-hls-integration ;
        # check-circt-hls-cosim fails one with 500s timeout
        #CIRCT-HLS cosim :: suites/Dynamatic/insertion_sort/tst_insertion_sort.c
        ninja check-circt-hls-cosim ;
)
