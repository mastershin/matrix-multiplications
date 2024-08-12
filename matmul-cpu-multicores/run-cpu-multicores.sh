#!/usr/bin/env bash

echo "Running CPU executables..."
# make run ARGS="--size m --loop 10"

if [ $@ ]; then
 echo "Running CPU executables with arguments: $@"
  ./matmul_cpu_multicores.ex $@
else
  echo "Running CPU executables with default arguments... (--size and --loop)"
  set -x
  ./matmul_cpu_multicores.ex --size l --loop 5
fi
