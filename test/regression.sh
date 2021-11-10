#!/usr/bin/env bash

# "regression" tests which just executes example runs which we expect to work.

set -e

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
cd $SCRIPT_DIR/..

## Statically scheduled
fud exec "examples/c/fir/fir.c"             \
  --from c                                  \
  --to mlir-scf-while                       \
  -s circt_hls.toplevel "fir"


### Polygeist to **mlir (calyx)**  
fud exec "examples/c/fir/fir.c"             \
  --from c                                  \
  --to mlir-calyx                           \
  -s circt_hls.toplevel "fir"


### Polygeist to **calyx**  
fud exec "examples/c/fir/fir.c"             \
  --from c                                  \
  --to futil                                \
  -s circt_hls.toplevel "fir"

## Dynamically scheduled
### Polygeist to **mlir (handshake)**  
fud exec "examples/c/fir/fir.c" \
  --from c                      \
  --to mlir-handshake           \
  -s circt_hls.toplevel fir


### Polygeist to **mlir (FIRRTL)**  
fud exec "examples/c/fir/fir.c" \
  --from c                      \
  --to mlir-firrtl              \
  -s circt_hls.toplevel fir


### Handshake to Verilog
fud exec ${handshake MLIR file} \
  --from mlir-handshake         \
  --to synth-verilog

### Handshake to synthesized
fud exec ${handshake MLIR file} \
  --from mlir-handshake         \
  --to synth-files -o ${outdir}
