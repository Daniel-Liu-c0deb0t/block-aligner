#![cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "wasm32"))]

use rand::prelude::*;

use bio::alignment::pairwise::*;
use bio::scores::blosum62;

use better_alignment::scan_minecraft::*;
use better_alignment::scores::*;
use better_alignment::simulate::*;

use std::str;

fn test(iter: usize, len: usize, k: usize) -> usize {
    let mut wrong = 0usize;
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
            println!(
                "bio: {}, ours: {}\nq: {}\nr: {}\nk: {}",
                bio_score,
                scan_score,
                str::from_utf8(&q).unwrap(),
                str::from_utf8(&r).unwrap(),
                k
            );
        }
    }

    wrong
}

fn main() {
    let iter = 100;
    let lens = [5, 10, 20, 50, 100];
    let rcp_ks = [10, 5];

    let mut total_wrong = 0usize;
    let mut total = 0usize;

    for &len in &lens {
        for &rcp_k in &rcp_ks {
            let wrong = test(iter, len, len / rcp_k);
            println!("\nlen: {}, 1/k: {}, iter: {}, wrong: {}\n", len, rcp_k, iter, wrong);
            total_wrong += wrong;
            total += iter;
        }
    }

    println!("\ntotal: {}, wrong: {}", total, total_wrong);
    println!("Done!");
}
