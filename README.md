## matmul-cpu
- Not optimized CPU based matrix multiplication

## matmul-cpu-multicores
- optimized for Multi-core CPU based matrix multiplication
- Uses <threads> c++ lib

## matmul-avx-x64
- Utilize Intel based AVX instruction
- Uses AVX (128 bit), AVX2 (256 bit) and AVX-512 (512 bit)

## matmul-cuda
- CUDA based

## matmul-neon-arm64
- ARM64 based
- Apple
- Uses Neon for similar to AVX operations

## matmul-rust
- Rust based (Not optimized, single core only)