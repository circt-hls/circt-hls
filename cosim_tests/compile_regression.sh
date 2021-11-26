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

echo "Running tests..."

# For each .c file in all subdirectories which is not a testbench
for c_file in $(find . -name "*.c"); do
    # if basename of file starts with tst_
    set c_file = basename $c_file
    if [[ $c_file != tst_* ]]; then
        # Run the dynamically scheduled path
        c_to_sv_dyn $c_file
        # Run the statically scheduled path
        c_to_sv_stat $c_file
    fi
done

echo "Done - all regression tests passed!"
