#![feature(test)]
#![cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "wasm32"))]

extern crate test;
use test::{Bencher, black_box};

use rand::prelude::*;

use better_alignment::scan_block::*;
use better_alignment::scores::*;
use better_alignment::simulate::*;

fn bench_scan_aa_core<const K: usize>(b: &mut Bencher, len: usize) {
    let mut rng = StdRng::seed_from_u64(1234);
    let r = black_box(rand_str(len, &AMINO_ACIDS, &mut rng));
    let q = black_box(rand_mutate(&r, K, &AMINO_ACIDS, &mut rng));
    let r = PaddedBytes::from_bytes(&r, 256, false);
    let q = PaddedBytes::from_bytes(&q, 256, false);
    type BenchParams = GapParams<-11, -1>;

    b.iter(|| {
        let a = Block::<BenchParams, _, 16, 256, false, false>::align(&q, &r, &BLOSUM62, 0, 10, 0);
        a.res()
    });
}

fn bench_scan_nuc_core<const K: usize>(b: &mut Bencher, len: usize) {
    let mut rng = StdRng::seed_from_u64(1234);
    let r = black_box(rand_str(len, &NUC, &mut rng));
    let q = black_box(rand_mutate(&r, K, &NUC, &mut rng));
    let r = PaddedBytes::from_bytes(&r, 256, true);
    let q = PaddedBytes::from_bytes(&q, 256, true);
    type BenchParams = GapParams<-1, -1>;

    b.iter(|| {
        let a = Block::<BenchParams, _, 16, 256, false, false>::align(&q, &r, &NW1, 0, 20, 0);
        a.res()
    });
}

#[bench]
fn bench_scan_aa_10_100(b: &mut Bencher) { bench_scan_aa_core::<10>(b, 100); }
#[bench]
fn bench_scan_aa_100_1000(b: &mut Bencher) { bench_scan_aa_core::<100>(b, 1000); }
#[bench]
fn bench_scan_aa_1000_10000(b: &mut Bencher) { bench_scan_aa_core::<1000>(b, 10000); }

#[bench]
fn bench_scan_nuc_100_1000(b: &mut Bencher) { bench_scan_nuc_core::<100>(b, 1000); }
#[bench]
fn bench_scan_nuc_1000_10000(b: &mut Bencher) { bench_scan_nuc_core::<1000>(b, 10000); }
