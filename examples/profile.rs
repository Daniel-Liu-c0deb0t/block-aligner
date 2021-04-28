#![cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "wasm32"))]

use rand::prelude::*;

use better_alignment::scan_block::*;
use better_alignment::scores::*;
use better_alignment::simulate::*;

fn run<const K: usize>(len: usize) -> AlignResult {
    let mut rng = StdRng::seed_from_u64(1234);
    let r = rand_str(len, &AMINO_ACIDS, &mut rng);
    let q = rand_mutate(&r, K, &AMINO_ACIDS, &mut rng);
    let r = PaddedBytes::from_bytes(&r, 16, false);
    let q = PaddedBytes::from_bytes(&q, 16, false);
    type RunParams = GapParams<-11, -1>;

    let a = Block::<RunParams, _, 16, 256, false, false>::align(&q, &r, &BLOSUM62, 0, 6);
    a.res()
}

fn main() {
    for _i in 0..10000 {
        run::<1000>(10000);
    }
}
