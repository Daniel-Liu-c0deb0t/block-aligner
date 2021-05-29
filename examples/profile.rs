#![feature(bench_black_box)]
#![cfg(any(
        all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"),
        all(target_arch = "wasm32", target_feature = "simd128")
))]

use rand::prelude::*;

use block_aligner::scan_block::*;
use block_aligner::scores::*;
use block_aligner::simulate::*;

use std::hint::black_box;

fn run(len: usize, k: usize) {
    let mut rng = StdRng::seed_from_u64(1234);
    let r = rand_str(len, &AMINO_ACIDS, &mut rng);
    let q = rand_mutate(&r, k, &AMINO_ACIDS, &mut rng);
    let r = PaddedBytes::from_bytes(&r, 2048, &BLOSUM62);
    let q = PaddedBytes::from_bytes(&q, 2048, &BLOSUM62);
    let run_gaps = Gaps { open: -11, extend: -1 };

    for _i in 0..10000 {
        let a = Block::<_, true, true>::align(&q, &r, &BLOSUM62, run_gaps, 32..=32, 1000);
        black_box(a.res());
    }
}

fn main() {
    run(10000, 1000);
}
