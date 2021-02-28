#![feature(test)]

extern crate test;
use test::{Bencher, black_box};

use rand::prelude::*;

use better_alignment::scan_avx2::*;
use better_alignment::scores::*;

static AMINO_ACIDS: [u8; 20] = [
    b'A', b'C', b'D', b'E', b'F', b'G', b'H', b'I', b'K', b'L',
    b'M', b'N', b'P', b'Q', b'R', b'S', b'T', b'V', b'W', b'Y'
];

static NUC: [u8; 5] = [
    b'A', b'C', b'G', b'N', b'T'
];

fn bench_scan_avx2_aa_core<const K: usize>(b: &mut Bencher, len: usize) {
    let mut rng = StdRng::seed_from_u64(1234);
    let r = black_box(rand_str(len, &AMINO_ACIDS, &mut rng));
    let q = black_box(rand_mutate(&r, K, &AMINO_ACIDS, &mut rng));
    type BenchParams = Params<-11, -1, 1024>;

    b.iter(|| {
        unsafe {
            let mut a = ScanAligner::<BenchParams, _, K, false>::new(&q, &BLOSUM62);
            a.align(&r);
            a.score()
        }
    });
}

fn bench_scan_avx2_nuc_core<const K: usize>(b: &mut Bencher, len: usize) {
    let mut rng = StdRng::seed_from_u64(1234);
    let r = black_box(rand_str(len, &NUC, &mut rng));
    let q = black_box(rand_mutate(&r, K, &NUC, &mut rng));
    type BenchParams = Params<-1, -1, 2048>;

    b.iter(|| {
        unsafe {
            let mut a = ScanAligner::<BenchParams, _, K, false>::new(&q, &NW1);
            a.align(&r);
            a.score()
        }
    });
}

#[bench]
fn bench_scan_avx2_aa_15_100(b: &mut Bencher) { bench_scan_avx2_aa_core::<15>(b, 100); }
#[bench]
fn bench_scan_avx2_aa_15_1000(b: &mut Bencher) { bench_scan_avx2_aa_core::<15>(b, 1000); }
#[bench]
fn bench_scan_avx2_aa_15_10000(b: &mut Bencher) { bench_scan_avx2_aa_core::<15>(b, 10000); }

#[bench]
fn bench_scan_avx2_aa_1023_10000(b: &mut Bencher) { bench_scan_avx2_aa_core::<1023>(b, 10000); }
#[bench]
fn bench_scan_avx2_aa_1024_10000(b: &mut Bencher) { bench_scan_avx2_aa_core::<1024>(b, 10000); }
#[bench]
fn bench_scan_avx2_aa_2500_5000(b: &mut Bencher) { bench_scan_avx2_aa_core::<2500>(b, 5000); }

#[bench]
fn bench_scan_avx2_nuc_1023_10000(b: &mut Bencher) { bench_scan_avx2_nuc_core::<1023>(b, 10000); }
#[bench]
fn bench_scan_avx2_nuc_1024_10000(b: &mut Bencher) { bench_scan_avx2_nuc_core::<1024>(b, 10000); }
#[bench]
fn bench_scan_avx2_nuc_2500_5000(b: &mut Bencher) { bench_scan_avx2_nuc_core::<2500>(b, 5000); }
#[bench]
fn bench_scan_avx2_nuc_5_100(b: &mut Bencher) { bench_scan_avx2_nuc_core::<5>(b, 100); }
#[bench]
fn bench_scan_avx2_nuc_50_1000(b: &mut Bencher) { bench_scan_avx2_nuc_core::<50>(b, 1000); }

fn rand_mutate<R: Rng>(a: &[u8], k: usize, alpha: &[u8], rng: &mut R) -> Vec<u8> {
    let mut edits = vec![0u8; a.len()];
    let curr_k: usize = rng.gen_range(k / 2..k + 1);
    let mut idx: Vec<usize> = (0usize..a.len()).collect();
    idx.shuffle(rng);

    for i in 0..curr_k {
        edits[idx[i]] = rng.gen_range(1u8..4u8);
    }

    let mut b = vec![];

    for i in 0..a.len() {
        match edits[i] {
            0u8 => { // same
                b.push(a[i]);
            },
            1u8 => { // diff
                let mut iter = alpha.choose_multiple(rng, 2);
                let first = *iter.next().unwrap();
                let second = *iter.next().unwrap();
                b.push(if first == a[i] { second } else { first });
            },
            2u8 => { // insert
                b.push(*alpha.choose(rng).unwrap());
                b.push(a[i]);
            },
            3u8 => (), // delete
            _ => panic!("This should not have been reached!")
        }
    }

    b
}

fn rand_str<R: Rng>(length: usize, alpha: &[u8], rng: &mut R) -> Vec<u8> {
    let mut res = vec![0u8; length];

    for i in 0..length {
        res[i] = *alpha.choose(rng).unwrap();
    }

    res
}
