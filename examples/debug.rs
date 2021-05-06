#![cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "wasm32"))]

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
    let r_padded = PaddedBytes::from_bytes(&r, 2048, false);
    let q_padded = PaddedBytes::from_bytes(&q, 2048, false);
    type RunParams = GapParams<-11, -1>;

    let block_aligner = Block::<RunParams, _, 16, 2048, true, false>::align(&q_padded, &r_padded, &BLOSUM62, 0, 6);
    let scan_score = block_aligner.res().score;
    let scan_cigar = block_aligner.trace().cigar();

    println!(
        "score: {}\nq: {}\nr: {}\ntrace: {}",
        scan_score,
        str::from_utf8(&q).unwrap(),
        str::from_utf8(&r).unwrap(),
        scan_cigar
    );
}
