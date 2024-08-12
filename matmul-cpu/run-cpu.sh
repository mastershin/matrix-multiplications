#!/usr/bin/env bash

if [ $@ ]; then
 echo "Running CPU executables with arguments: $@"
  ./matmul_cpu.ex $@
else
  echo "Running CPU executables with default arguments... (--size and --loop)"
  set -x
  ./matmul_cpu.ex --size l --loop 5
fi



