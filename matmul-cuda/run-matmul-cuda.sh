#!/usr/bin/env bash

./matmul_cuda.ex --size s --loop 200

./matmul_cuda.ex --size m --loop 200

./matmul_cuda.ex --size l --loop 200
