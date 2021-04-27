#![cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "wasm32"))]

use better_alignment::scan_block::*;
use better_alignment::scores::*;

use std::{env, str};

fn main() {
    let mut args = env::args().skip(1);
    let mut q = args.next().unwrap();
    q.make_ascii_uppercase();
    let q = q.as_bytes().to_owned();
    let mut r = args.next().unwrap();
    r.make_ascii_uppercase();
    let r = r.as_bytes().to_owned();
    let r_padded = PaddedBytes::from_bytes(&r, 256, false);
    let q_padded = PaddedBytes::from_bytes(&q, 256, false);
    type RunParams = GapParams<-11, -1>;

    let block_aligner = Block::<RunParams, _, 16, 256, false, false>::align(&q_padded, &r_padded, &BLOSUM62, 0, 7, 0);
    let scan_score = block_aligner.res().score;

    println!(
        "score: {},\nq: {},\nr: {}",
        scan_score,
        str::from_utf8(&q).unwrap(),
        str::from_utf8(&r).unwrap()
    );
}
