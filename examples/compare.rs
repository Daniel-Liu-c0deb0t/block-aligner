#![cfg(any(
        all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"),
        all(target_arch = "wasm32", target_feature = "simd128")
))]

use block_aligner::scan_block::*;
use block_aligner::scores::*;

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn test(file_name: &str, max_size: usize) -> (usize, usize, f64, usize, f64) {
    let reader = BufReader::new(File::open(file_name).unwrap());
    let mut count = 0;
    let mut other_better = 0;
    let mut other_better_avg = 0f64;
    let mut us_better = 0;
    let mut us_better_avg = 0f64;

    for line in reader.lines() {
        let line = line.unwrap();
        let mut row = line.split_ascii_whitespace().take(5);
        let q = row.next().unwrap().to_ascii_uppercase();
        let r = row.next().unwrap().to_ascii_uppercase();
        let other_score = row.next().unwrap().parse::<i32>().unwrap();
        let _other_i = row.next().unwrap().parse::<usize>().unwrap();
        let _other_j = row.next().unwrap().parse::<usize>().unwrap();

        //let x_drop = 100;
        let x_drop = 50;
        //let matrix = NucMatrix::new_simple(2, -3);
        let matrix = NucMatrix::new_simple(1, -1);
        let r_padded = PaddedBytes::from_bytes(r.as_bytes(), 2048, &matrix);
        let q_padded = PaddedBytes::from_bytes(q.as_bytes(), 2048, &matrix);
        //let run_gaps = Gaps { open: -5, extend: -1 };
        let run_gaps = Gaps { open: -2, extend: -1 };

        // ours
        let block_aligner = Block::<_, true, true>::align(&q_padded, &r_padded, &matrix, run_gaps, max_size..=max_size, x_drop);
        let scan_res = block_aligner.res();
        let scan_score = scan_res.score;

        if scan_score > other_score {
            us_better += 1;
            us_better_avg += ((scan_score - other_score) as f64) / (scan_score as f64);
        }

        if scan_score < other_score {
            other_better += 1;
            other_better_avg += ((other_score - scan_score) as f64) / (other_score as f64);
        }

        count += 1;
    }

    (count, other_better, other_better_avg / (other_better as f64), us_better, us_better_avg / (us_better as f64))
}

fn main() {
    let other_file = env::args().skip(1).next().expect("Pass in the path to a tab-separated file to compare to!");
    let max_sizes = [32, 64];

    for &max_size in &max_sizes {
        println!("\nmax size: {}", max_size);

        let (count, other_better, other_better_avg, us_better, us_better_avg) = test(&other_file, max_size);

        println!(
            "\ntotal: {}, other better: {}, avg: {}, us better: {}, avg: {}",
            count,
            other_better,
            other_better_avg,
            us_better,
            us_better_avg
        );
    }

    println!("Done!");
}
