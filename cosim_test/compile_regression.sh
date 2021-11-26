#!/usr/bin/env bash

# "regression" tests which just executes example runs which we expect to work.

set -e

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
cd $SCRIPT_DIR

# Create directory for test results
resultDir=${SCRIPT_DIR}/out
mkdir -p $resultDir

## End-to-end dynamically scheduled path
c_to_sv_dyn() {
    # The first argument of this function is a full path to a file;
    # get the basename of the file without the extension.
    local basename=${1##*/}
    basename=${basename%.*}

    outdir=$resultDir/$basename
    mkdir -p $outdir

    printf "= Dynamically scheduled HLS'ing $1...\n"
    fud exec --from c --to mlir-affine -o $outdir/${basename}_affine.mlir $1
    printf "\tLowered to affine! (Polygeist)...!"
    fud exec --from mlir-affine --to mlir-scf-while -o $outdir/${basename}_scf.mlir $outdir/${basename}_affine.mlir
    printf "\r\tLowered to scf...!"
    fud exec --from mlir-scf-while --to mlir-handshake-buffered -o $outdir/${basename}_handshake.mlir $outdir/${basename}_scf.mlir
    printf "\r\tLowered to handshake...!"
    fud exec --from mlir-handshake-buffered --to mlir-firrtl -o $outdir/${basename}_firrtl.mlir $outdir/${basename}_handshake.mlir
    printf "\r\tLowered to FIRRTL...!"
    fud exec --from mlir-firrtl --to synth-verilog -o $outdir/${basename}_sv.sv $outdir/${basename}_firrtl.mlir
}

## End-to-end statically scheduled path
c_to_sv_stat() {
    printf "= Statically scheduled HLS'ing $1\n"
    fud exec --from c --to synth-verilog --through mlir-calyx $1
}

echo "Running tests..."

# For each .c file in all subdirectories which is not a testbench
for c_file in $(find . -name "*.c"); do
    # if basename of file starts with tst_
    c_file_base=$(basename $c_file)
    if [[ $c_file_base != "tst_"* ]]; then
        # Run the dynamically scheduled path
        c_to_sv_dyn $c_file
        # Run the statically scheduled path
        # c_to_sv_stat $c_file
    fi
done

echo "Done - all regression tests passed!"
