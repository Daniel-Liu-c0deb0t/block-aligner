#![feature(test)]
#![cfg(any(
        all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"),
        all(target_arch = "wasm32", target_feature = "simd128")
))]

extern crate test;
use test::{Bencher, black_box};

use rand::prelude::*;

use bio::alignment::pairwise::*;
use bio::scores::blosum62;

#[cfg(not(target_arch = "wasm32"))]
use bio::alignment::distance::simd::bounded_levenshtein;

#[cfg(not(target_arch = "wasm32"))]
use parasailors::{Matrix, *};

use block_aligner::scan_block::*;
use block_aligner::scores::*;
use block_aligner::simulate::*;

fn bench_rustbio_aa_core<const K: usize>(b: &mut Bencher, len: usize) {
    let mut rng = StdRng::seed_from_u64(1234);
    let r = black_box(rand_str(len, &AMINO_ACIDS, &mut rng));
    let q = black_box(rand_mutate(&r, K, &AMINO_ACIDS, &mut rng));

    b.iter(|| {
        let mut bio_aligner = Aligner::with_capacity(q.len(), r.len(), -10, -1, &blosum62);
        bio_aligner.global(&q, &r).score
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn bench_parasailors_aa_core<const K: usize>(b: &mut Bencher, len: usize) {
    let mut rng = StdRng::seed_from_u64(1234);
    let r = black_box(rand_str(len, &AMINO_ACIDS, &mut rng));
    let q = black_box(rand_mutate(&r, K, &AMINO_ACIDS, &mut rng));
    let matrix = Matrix::new(MatrixType::Blosum62);
    let profile = Profile::new(&q, &matrix);

    b.iter(|| {
        global_alignment_score(&profile, &r, 11, 1)
    });
}

fn bench_scan_aa_core<const K: usize>(b: &mut Bencher, len: usize, insert: bool) {
    let mut rng = StdRng::seed_from_u64(1234);
    let r = black_box(rand_str(len, &AMINO_ACIDS, &mut rng));
    let q = if insert {
        black_box(rand_mutate_insert(&r, K, &AMINO_ACIDS, len / 10, &mut rng))
    } else {
        black_box(rand_mutate(&r, K, &AMINO_ACIDS, &mut rng))
    };
    let r = PaddedBytes::from_bytes(&r, 2048, false);
    let q = PaddedBytes::from_bytes(&q, 2048, false);
    type BenchParams = GapParams<-11, -1>;

    b.iter(|| {
        let a = Block::<BenchParams, _, 32, 2048, false, false>::align(&q, &r, &BLOSUM62, 0, 8);
        a.res()
    });
}

fn bench_scan_aa_core_small<const K: usize>(b: &mut Bencher, len: usize) {
    let mut rng = StdRng::seed_from_u64(1234);
    let r = black_box(rand_str(len, &AMINO_ACIDS, &mut rng));
    let q = black_box(rand_mutate(&r, K, &AMINO_ACIDS, &mut rng));
    let r = PaddedBytes::from_bytes(&r, 2048, false);
    let q = PaddedBytes::from_bytes(&q, 2048, false);
    type BenchParams = GapParams<-11, -1>;

    b.iter(|| {
        let a = Block::<BenchParams, _, 16, 16, false, false>::align(&q, &r, &BLOSUM62, 0, 8);
        a.res()
    });
}

fn bench_scan_aa_core_trace<const K: usize>(b: &mut Bencher, len: usize) {
    let mut rng = StdRng::seed_from_u64(1234);
    let r = black_box(rand_str(len, &AMINO_ACIDS, &mut rng));
    let q = black_box(rand_mutate(&r, K, &AMINO_ACIDS, &mut rng));
    let r = PaddedBytes::from_bytes(&r, 2048, false);
    let q = PaddedBytes::from_bytes(&q, 2048, false);
    type BenchParams = GapParams<-11, -1>;

    b.iter(|| {
        let a = Block::<BenchParams, _, 32, 2048, true, false>::align(&q, &r, &BLOSUM62, 0, 8);
        //a.res()
        (a.res(), a.trace().cigar(q.len(), r.len()))
    });
}

fn bench_scan_nuc_core<const K: usize>(b: &mut Bencher, len: usize) {
    let mut rng = StdRng::seed_from_u64(1234);
    let r = black_box(rand_str(len, &NUC, &mut rng));
    let q = black_box(rand_mutate(&r, K, &NUC, &mut rng));
    let r = PaddedBytes::from_bytes(&r, 2048, true);
    let q = PaddedBytes::from_bytes(&q, 2048, true);
    type BenchParams = GapParams<-1, -1>;

    b.iter(|| {
        let a = Block::<BenchParams, _, 32, 2048, false, false>::align(&q, &r, &NW1, 0, 8);
        a.res()
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn bench_triple_accel_core<const K: usize>(b: &mut Bencher, len: usize) {
    let mut rng = StdRng::seed_from_u64(1234);
    let r = black_box(rand_str(len, &NUC, &mut rng));
    let q = black_box(rand_mutate(&r, K, &NUC, &mut rng));

    b.iter(|| {
        bounded_levenshtein(&q, &r, K as u32)
    });
}

#[bench]
fn bench_scan_aa_10_100(b: &mut Bencher) { bench_scan_aa_core::<10>(b, 100, false); }
#[bench]
fn bench_scan_aa_100_1000(b: &mut Bencher) { bench_scan_aa_core::<100>(b, 1000, false); }
#[bench]
fn bench_scan_aa_1000_10000(b: &mut Bencher) { bench_scan_aa_core::<1000>(b, 10000, false); }

#[bench]
fn bench_scan_aa_10_100_insert(b: &mut Bencher) { bench_scan_aa_core::<10>(b, 100, true); }
#[bench]
fn bench_scan_aa_100_1000_insert(b: &mut Bencher) { bench_scan_aa_core::<100>(b, 1000, true); }
#[bench]
fn bench_scan_aa_1000_10000_insert(b: &mut Bencher) { bench_scan_aa_core::<1000>(b, 10000, true); }

#[bench]
fn bench_scan_aa_10_100_small(b: &mut Bencher) { bench_scan_aa_core_small::<10>(b, 100); }
#[bench]
fn bench_scan_aa_100_1000_small(b: &mut Bencher) { bench_scan_aa_core_small::<100>(b, 1000); }
#[bench]
fn bench_scan_aa_1000_10000_small(b: &mut Bencher) { bench_scan_aa_core_small::<1000>(b, 10000); }

#[bench]
fn bench_scan_aa_10_100_trace(b: &mut Bencher) { bench_scan_aa_core_trace::<10>(b, 100); }
#[bench]
fn bench_scan_aa_100_1000_trace(b: &mut Bencher) { bench_scan_aa_core_trace::<100>(b, 1000); }
#[bench]
fn bench_scan_aa_1000_10000_trace(b: &mut Bencher) { bench_scan_aa_core_trace::<1000>(b, 10000); }

#[bench]
fn bench_scan_nuc_100_1000(b: &mut Bencher) { bench_scan_nuc_core::<100>(b, 1000); }
#[bench]
fn bench_scan_nuc_1000_10000(b: &mut Bencher) { bench_scan_nuc_core::<1000>(b, 10000); }

#[cfg(not(target_arch = "wasm32"))]
#[bench]
fn bench_triple_accel_100_1000(b: &mut Bencher) { bench_triple_accel_core::<100>(b, 1000); }
#[cfg(not(target_arch = "wasm32"))]
#[bench]
fn bench_triple_accel_1000_10000(b: &mut Bencher) { bench_triple_accel_core::<1000>(b, 10000); }

#[bench]
fn bench_rustbio_aa_10_100(b: &mut Bencher) { bench_rustbio_aa_core::<10>(b, 100); }
#[bench]
fn bench_rustbio_aa_100_1000(b: &mut Bencher) { bench_rustbio_aa_core::<100>(b, 1000); }

#[cfg(not(target_arch = "wasm32"))]
#[bench]
fn bench_parasailors_aa_10_100(b: &mut Bencher) { bench_parasailors_aa_core::<10>(b, 100); }
#[cfg(not(target_arch = "wasm32"))]
#[bench]
fn bench_parasailors_aa_100_1000(b: &mut Bencher) { bench_parasailors_aa_core::<100>(b, 1000); }
#[cfg(not(target_arch = "wasm32"))]
#[bench]
fn bench_parasailors_aa_1000_10000(b: &mut Bencher) { bench_parasailors_aa_core::<1000>(b, 10000); }
