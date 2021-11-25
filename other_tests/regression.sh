#!/usr/bin/env bash

# "regression" tests which just executes example runs which we expect to work.

set -e

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
cd $SCRIPT_DIR/..

# Create directory for test results
resultDir=${SCRIPT_DIR}/results
mkdir -p $resultDir

## End-to-end dynamically scheduled path
c_to_sv_dyn() {
    # The first argument of this function is a full path to a file;
    # get the basename of the file without the extension.
    local basename=${1##*/}
    basename=${basename%.*}

    printf "= Dynamically scheduled HLS'ing $1...\n"
    fud exec --from c --to mlir-scf-while -o $resultDir/${basename}_scf.mlir $1 --verbose
    printf "\tLowered to scf...!\n"
    fud exec --from mlir-scf-while --to mlir-handshake-buffered -o $resultDir/${basename}_handshake.mlir $resultDir/${basename}_scf.mlir --verbose
    printf "\tLowered to handshake...!\n"
    fud exec --from mlir-handshake --to synth-verilog -o $resultDir/${basename}_sv.sv $resultDir/${basename}_handshake.mlir --verbose
    printf "\tLowered to verilog...!\n"
}

## End-to-end statically scheduled path
c_to_sv_stat() {
    printf "= Statically scheduled HLS'ing $1\n"
    fud exec --from c --to synth-verilog --through mlir-calyx $1
}

Filelist=$SCRIPT_DIR/filelist.lst

echo "Iterating over files in $Filelist"
while read fn; do
    c_to_sv_dyn ${fn}
done <$Filelist

echo "Done - all regression tests passed!"
