#![cfg(any(
        all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"),
        all(target_arch = "wasm32", target_feature = "simd128")
))]

use bio::alignment::pairwise::*;
use bio::scores::blosum62;

use block_aligner::scan_block::*;
use block_aligner::scores::*;

use std::{env, str};

fn main() {
    let mut args = env::args().skip(1);
    let mut q = args.next().unwrap();
    q.make_ascii_uppercase();
    let q = q.as_bytes().to_owned();
    let mut r = args.next().unwrap();
    r.make_ascii_uppercase();
    let r = r.as_bytes().to_owned();
    let r_padded = PaddedBytes::from_bytes::<AAMatrix>(&r, 2048);
    let q_padded = PaddedBytes::from_bytes::<AAMatrix>(&q, 2048);
    let run_gaps = Gaps { open: -11, extend: -1 };

    let mut bio_aligner = Aligner::with_capacity(q.len(), r.len(), -10, -1, &blosum62);
    let bio_alignment = bio_aligner.global(&q, &r);
    let bio_score = bio_alignment.score;

    let block_aligner = Block::<_, true, false>::align(&q_padded, &r_padded, &BLOSUM62, run_gaps, 32..=256, 0);
    let scan_score = block_aligner.res().score;
    let scan_cigar = block_aligner.trace().cigar(q.len(), r.len());
    let (a, b) = scan_cigar.format(&q, &r);

    println!(
        "bio: {}\nours: {}\nq (len = {}): {}\nr (len = {}): {}\nour trace: {}\nour pretty:\n{}\n{}\nbio pretty:\n{}",
        bio_score,
        scan_score,
        q.len(),
        str::from_utf8(&q).unwrap(),
        r.len(),
        str::from_utf8(&r).unwrap(),
        scan_cigar,
        a,
        b,
        bio_alignment.pretty(&q, &r)
    );
}
