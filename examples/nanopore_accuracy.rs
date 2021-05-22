#![cfg(any(
        all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"),
        all(target_arch = "wasm32", target_feature = "simd128")
))]

#[cfg(not(target_arch = "wasm32"))]
use parasailors::{Matrix, *};

use block_aligner::scan_block::*;
use block_aligner::scores::*;

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn test(file_name: &str, verbose: bool) -> (usize, f64, usize) {
    let mut wrong = 0;
    let mut wrong_avg = 0;
    let mut count = 0;
    let reader = BufReader::new(File::open(file_name).unwrap());
    let all_lines = reader.lines().collect::<Vec<_>>();

    for lines in all_lines.chunks(2) {
        let r = lines[0].as_ref().unwrap().to_ascii_uppercase();
        let q = lines[1].as_ref().unwrap().to_ascii_uppercase();

        // parasail
        let matrix = Matrix::new(MatrixType::IdentityWithPenalty);
        let profile = Profile::new(q.as_bytes(), &matrix);
        let parasail_score = global_alignment_score(&profile, r.as_bytes(), 2, 1);

        let r_padded = PaddedBytes::from_bytes(r.as_bytes(), 2048, &NW1);
        let q_padded = PaddedBytes::from_bytes(q.as_bytes(), 2048, &NW1);
        type RunParams = GapParams<-2, -1>;

        // ours
        let block_aligner = Block::<RunParams, _, false, false>::align(&q_padded, &r_padded, &NW1, 32..=2048, 0);
        let scan_score = block_aligner.res().score;

        if parasail_score != scan_score {
            wrong += 1;
            wrong_avg += (parasail_score - scan_score) as i64;

            if verbose {
                println!(
                    "parasail: {}, ours: {}\nq (len = {}): {}\nr (len = {}): {}",
                    parasail_score,
                    scan_score,
                    q.len(),
                    q,
                    r.len(),
                    r
                );
            }
        }

        count += 1;
    }

    (wrong, (wrong_avg as f64) / (count as f64), count)
}

fn main() {
    let arg1 = env::args().skip(1).next();
    let verbose = arg1.is_some() && arg1.unwrap() == "-v";
    let (wrong, wrong_avg, count) = test("data/supplementary_data/sequences.txt", verbose);
    println!("\ntotal: {}, wrong: {}, wrong avg: {}", count, wrong, wrong_avg);
    println!("Done!");
}
