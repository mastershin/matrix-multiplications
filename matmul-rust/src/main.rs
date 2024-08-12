extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::env;
use std::io::Write;
use std::time::Instant;
use std::collections::HashMap;

// Classic CPU matrix multiplication using for loop (slow)
#[allow(non_snake_case)]
fn matmul_cpu(A: &Array2<f32>, B: &Array2<f32>, C: &mut Array2<f32>, m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            C[[i, j]] = 0.0;
            for p in 0..k {
                C[[i, j]] += A[[i, p]] * B[[p, j]];
            }
        }
    }
}

#[allow(non_snake_case)]
fn initialize_data(m: usize, n: usize, k: usize) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let A = Array2::random((m, k), Uniform::new(-1.0, 1.0));
    let B = Array2::random((k, n), Uniform::new(-1.0, 1.0));
    let C = Array2::<f32>::zeros((m, n));
    (A, B, C)
}

fn get_small_matrix_size() -> (usize, usize, usize) {
    (200, 150, 100)
}

fn get_medium_matrix_size() -> (usize, usize, usize) {
    (500, 300, 200)
}

fn get_large_matrix_size() -> (usize, usize, usize) {
    (4096, 1024, 1024)
}

type ArgsMap = HashMap<String, String>;

fn parse_command_args() -> ArgsMap {
    let args: Vec<String> = env::args().collect();
    let mut args_map = ArgsMap::new();

    let mut i = 1;
    while i < args.len() {
        let arg = &args[i];
        if arg.starts_with("--") {
            let key = arg[2..].to_string();
            if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                args_map.insert(key, args[i + 1].to_string());
                i += 1;
            } else {
                args_map.insert(key, String::new());
            }
        }
        i += 1;
    }
    args_map
}

fn die(msg: &str) -> ! {
    eprintln!("{}", msg);
    std::process::exit(1);
}

fn process_commands(args_map: &ArgsMap) -> (usize, usize, usize, usize) {
    let matrix_size = args_map.get("size").unwrap_or_else(|| die("Size argument is required. Use '--size s|m|l'."));

    let (m, n, k) = match matrix_size.as_str() {
        "s" => get_small_matrix_size(),
        "m" => get_medium_matrix_size(),
        "l" => get_large_matrix_size(),
        _ => die("Invalid size argument. Use 's', 'm', 'l'."),
    };

    let num_loops = match args_map.get("loop") {
        Some(loop_str) => match loop_str.parse::<usize>() {
            Ok(num) if num > 0 => num,
            _ => die("Invalid loop argument. It should be a positive integer."),
        },
        None => die("Loop argument is required. Use '--loop <positive integer>'."),
    };

    println!("Number of loops: {}", num_loops);

    (m, n, k, num_loops)
}

#[allow(non_snake_case)]
fn main() {
    let args_map = parse_command_args();
    if args_map.len() < 2 {
        eprintln!("Usage: ./main --size [s|m|l] --loop [num_loops]");
        std::process::exit(1);
    }

    let (m, n, k, num_loops) = process_commands(&args_map);

    println!(
        "Matrix Multiplication: A({}x{}) * B({}x{}) = C({}x{})",
        m, k, k, n, m, n
    );

    // Allocate memory for matrices A, B, and C
    let (A, B, mut C) = initialize_data(m, n, k);

    // Perform CPU matrix multiplication for verification
    matmul_cpu(&A, &B, &mut C, m, n, k);

    let start_cpu = Instant::now();

    println!("Starting the loop...");
    for _ in 0..num_loops {
        print!(".");
        std::io::stdout().flush().unwrap();
        matmul_cpu(&A, &B, &mut C, m, n, k);
    }
    let end_cpu = Instant::now();
    let duration = end_cpu.duration_since(start_cpu);

    println!();
    println!("CPU time: {:.2} seconds", duration.as_secs_f64());

    let sum: f32 = C.sum();
    println!("Sum: {}", sum);
}
