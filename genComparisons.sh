#!/bin/bash

# Executes comparison HLS runs for each of the supported examples included in
# this repository.

comparisons=(
  "$(pwd)/examples/c/triangular"
)

for d in ${comparisons[*]}; do
  python3 comparisondriver.py \
    --vivado-hls              \
    --circt-static            \
    --circt-dynamic           \
    $d
done