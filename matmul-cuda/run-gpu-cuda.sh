#!/usr/bin/env bash

./matmul_gpu.ex --size s --loop 100

./matmul_gpu.ex --size m --loop 100

./matmul_gpu.ex --size l --loop 100

./matmul_gpu.ex --size xl --loop 100

./matmul_gpu.ex --size xxl --loop 100
