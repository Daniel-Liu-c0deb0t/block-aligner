#![cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "wasm32"))]

use rand::prelude::*;

use bio::alignment::pairwise::*;
use bio::scores::blosum62;

use better_alignment::scan_minecraft::*;
use better_alignment::scores::*;
use better_alignment::simulate::*;

use std::{env, str, cmp};

fn test(iter: usize, len: usize, k: usize, verbose: bool) -> (usize, f64, i32, i32) {
    let mut wrong = 0usize;
    let mut wrong_avg = 0i64;
    let mut wrong_min = i32::MAX;
    let mut wrong_max = i32::MIN;
    let mut rng = StdRng::seed_from_u64(1234);

    for _i in 0..iter {
        let r = rand_str(len, &AMINO_ACIDS, &mut rng);
        let q = rand_mutate(&r, k, &AMINO_ACIDS, &mut rng);

        // rust-bio
        let mut bio_aligner = Aligner::with_capacity(q.len(), r.len(), -10, -1, &blosum62);
        let bio_score = bio_aligner.global(&q, &r).score;

        let r_padded = PaddedBytes::from_bytes(&r);
        let q_padded = PaddedBytes::from_bytes(&q);
        type RunParams = GapParams<-11, -1>;

        // ours
        let block_aligner = Block::<RunParams, _, false, false>::align(&q_padded, &r_padded, &BLOSUM62, 0);
        let scan_score = block_aligner.res().score;

        if bio_score != scan_score {
            wrong += 1;
            let score_diff = bio_score - scan_score;
            wrong_avg += score_diff as i64;
            wrong_min = cmp::min(wrong_min, score_diff);
            wrong_max = cmp::max(wrong_max, score_diff);

            if verbose {
                println!(
                    "bio: {}, ours: {}\nq: {}\nr: {}",
                    bio_score,
                    scan_score,
                    str::from_utf8(&q).unwrap(),
                    str::from_utf8(&r).unwrap()
                );
            }
        }
    }

    (wrong, (wrong_avg as f64) / (iter as f64), wrong_min, wrong_max)
}

fn main() {
    let arg1 = env::args().skip(1).next();
    let verbose = arg1.is_some() && arg1.unwrap() == "-v";
    let iter = 1000;
    let lens = [10, 20, 50, 100, 1000];
    let rcp_ks = [10, 5, 2];

    let mut total_wrong = 0usize;
    let mut total = 0usize;

    for &len in &lens {
        for &rcp_k in &rcp_ks {
            let (wrong, wrong_avg, wrong_min, wrong_max) = test(iter, len, len / rcp_k, verbose);
            println!(
                "\nlen: {}, k: {}, iter: {}, wrong: {}, wrong avg: {}, wrong min: {}, wrong max: {}\n",
                len,
                len / rcp_k,
                iter,
                wrong,
                wrong_avg,
                wrong_min,
                wrong_max
            );
            total_wrong += wrong;
            total += iter;
        }
    }

    println!("\ntotal: {}, wrong: {}", total, total_wrong);
    println!("Done!");
}
