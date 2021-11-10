#!/usr/bin/env bash

# Run this script in the circt-hls folder. Ensure that you've already built
# Polygeist using the instructions in README.md.
# This script takes care of the Calyx setup.

# Building calyx
pushd Calyx
cargo build

# Building the Calyx driver 'fud'. This will be installed in your Python
# executeables folder, so make sure that this folder is in your PATH.
pip3 install flit
flit -f fud/pyproject.toml install -s

# Configure fud
fud config global.futil_directory $(pwd)/target/debug/futil
fud check

echo "If anything relevant is missing from 'fud check', make sure you have the corresponding application installed and available in your path."


# Setting up the HLS 'fud' stages
# NOTE: modify these paths to point to your own CIRCT/MLIR builds
popd
fud register polygeist -p $(pwd)/stages/Polygeist/stage.py
fud config stages.polygeist.exec $(pwd)/Polygeist/build/bin/mlir-clang

fud register circt -p $(pwd)/stages/CIRCT/stage.py
fud config stages.circt.bin_dir $(pwd)/../circt/build/bin

fud register mlir -p $(pwd)/stages/MLIR/stage.py
fud config stages.mlir.bin_dir $(pwd)/../circt/llvm/build/bin
fud config stages.circt_hls.toplevel ""
fud config

# et voilà!
