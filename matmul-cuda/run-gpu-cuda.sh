#!/usr/bin/env bash

#./matmul_gpu.ex --size s --loop 200
#./matmul_gpu.ex --size m --loop 200
#./matmul_gpu.ex --size l --loop 200
#./matmul_gpu.ex --size xl --loop 200
#./matmul_gpu.ex --size xxl --loop 200

./matmul_gpu.ex --m 16384 --n 16384 --k 16384 --loop 200
