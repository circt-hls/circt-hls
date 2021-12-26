#!/bin/bash

# If first argument to the script is 'enabled', then run tests
if [ "$1" = "enabled" ]; then
  # Run tests
  echo "Running tests..."
  cd build
  ninja check-circt-hls
  ninja check-circt-hls-integration
  ninja check-circt-hls-integration-extended
else
  echo "Skipping tests..."
fi
