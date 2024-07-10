#!/usr/bin/env bash

# ./target/release/matmul $@

# example:
# cargo run --release -- --size m --loop 100

cargo run --release -- $@

