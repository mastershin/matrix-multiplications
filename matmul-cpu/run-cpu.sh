#!/usr/bin/env bash
# Reference: https://github.com/nvidia/cuda-samples

# Check for g++ compiler
if ! command -v g++ >/dev/null 2>&1; then
  echo "g++ not found. Please install it using your package manager."
  echo "For example, on Ubuntu/Debian:"
  echo "  sudo apt update"
  echo "  sudo apt install build-essential"
  exit 1
fi

echo "g++ found. Compiling CPU code..."
# g++ --std=c++17 main.cpp matmul_cpu.cpp -o matmul_cpu.ex
make

# Run the CPU and GPU executables
echo "Running CPU executables..."
# make run ARGS="--size m --loop 10"
./matmul_cpu.ex $@


