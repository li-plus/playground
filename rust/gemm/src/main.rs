use rand::Rng;

#[allow(non_snake_case)]
fn gemm(A: &[f32], B: &[f32], C: &mut [f32], M: usize, N: usize, K: usize) {
    for m in 0..M {
        for n in 0..N {
            let mut sum = 0.0;
            for k in 0..K {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

#[allow(non_snake_case)]
fn main() {
    const M: usize = 512;
    const N: usize = 512;
    const K: usize = 512;

    let warmup = 2;
    let active = 10;

    let mut rng = rand::rng();

    let mut A = Vec::with_capacity(M * K);
    for _ in 0..M * K {
        A.push(rng.random::<f32>());
    }

    let mut B = Vec::with_capacity(K * N);
    for _ in 0..K * N {
        B.push(rng.random::<f32>());
    }

    let mut C = vec![0.0; M * N];

    // warmup
    for _ in 0..warmup {
        gemm(&A, &B, &mut C, M, N, K);
    }

    // benchmark
    let start = std::time::Instant::now();
    for _ in 0..active {
        gemm(&A, &B, &mut C, M, N, K);
    }
    let duration = start.elapsed() / active;
    let gflops = (2 * M * N * K) as f64 / duration.as_secs_f64() * 1e-9;

    println!("GEMM duration: {duration:?}, {gflops} GFLOPS");
}
