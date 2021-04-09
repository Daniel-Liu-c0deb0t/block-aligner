#![feature(test)]
#![cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "wasm32"))]

extern crate test;
use test::{Bencher, black_box};

use rand::prelude::*;

use better_alignment::scan_minecraft::*;
use better_alignment::scores::*;

static AMINO_ACIDS: [u8; 20] = [
    b'A', b'C', b'D', b'E', b'F', b'G', b'H', b'I', b'K', b'L',
    b'M', b'N', b'P', b'Q', b'R', b'S', b'T', b'V', b'W', b'Y'
];

static NUC: [u8; 5] = [
    b'A', b'C', b'G', b'N', b'T'
];

fn bench_scan_aa_core<const K: usize>(b: &mut Bencher, len: usize) {
    let mut rng = StdRng::seed_from_u64(1234);
    let r = black_box(rand_str(len, &AMINO_ACIDS, &mut rng));
    let q = black_box(rand_mutate(&r, K, &AMINO_ACIDS, &mut rng));
    let r = PaddedBytes::from_bytes(&r);
    let q = PaddedBytes::from_bytes(&q);
    type BenchParams = GapParams<-11, -1>;

    b.iter(|| {
        let a = Block::<BenchParams, _, false, false>::align(&q, &r, &BLOSUM62, 0);
        a.res()
    });
}

fn bench_scan_nuc_core<const K: usize>(b: &mut Bencher, len: usize) {
    let mut rng = StdRng::seed_from_u64(1234);
    let r = black_box(rand_str(len, &NUC, &mut rng));
    let q = black_box(rand_mutate(&r, K, &NUC, &mut rng));
    let r = PaddedBytes::from_bytes(&r);
    let q = PaddedBytes::from_bytes(&q);
    type BenchParams = GapParams<-1, -1>;

    b.iter(|| {
        let a = Block::<BenchParams, _, false, false>::align(&q, &r, &NW1, 0);
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
