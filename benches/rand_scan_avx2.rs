#![feature(test)]
#![feature(min_const_generics)]

extern crate test;
use test::{Bencher, black_box};

use rand::prelude::*;

use better_alignment::scan_avx2::*;
use better_alignment::scores::*;

fn bench_scan_avx2_core<const K: usize>(b: &mut Bencher, len: usize) {
    let mut rng = StdRng::seed_from_u64(1234);
    let r = black_box(rand_protein(len, &mut rng));
    let q = black_box(rand_mutate(&r, K, &mut rng));
    type Scores = Gap<-11, -1>;

    b.iter(|| {
        unsafe { scan_align::<Scores, K, false, false>(&r, &q, &BLOSUM62) }
    });
}

#[bench]
fn bench_scan_avx2_15_100(b: &mut Bencher) { bench_scan_avx2_core::<15>(b, 100); }
#[bench]
fn bench_scan_avx2_15_1000(b: &mut Bencher) { bench_scan_avx2_core::<15>(b, 1000); }
#[bench]
fn bench_scan_avx2_15_10000(b: &mut Bencher) { bench_scan_avx2_core::<15>(b, 10000); }

#[bench]
fn bench_scan_avx2_1023_10000(b: &mut Bencher) { bench_scan_avx2_core::<1023>(b, 10000); }
#[bench]
fn bench_scan_avx2_1024_10000(b: &mut Bencher) { bench_scan_avx2_core::<1024>(b, 10000); }
#[bench]
fn bench_scan_avx2_2500_5000(b: &mut Bencher) { bench_scan_avx2_core::<2500>(b, 5000); }

static AMINO_ACIDS: [u8; 20] = [
    b'A', b'C', b'D', b'E', b'F', b'G', b'H', b'I', b'K', b'L',
    b'M', b'N', b'P', b'Q', b'R', b'S', b'T', b'V', b'W', b'Y'
];

fn rand_mutate<R: Rng>(a: &[u8], k: usize, rng: &mut R) -> Vec<u8> {
    let mut edits = vec![0u8; a.len()];
    let curr_k: usize = rng.gen_range(k / 2, k + 1);
    let mut idx: Vec<usize> = (0usize..a.len()).collect();
    idx.shuffle(rng);

    for i in 0..curr_k {
        edits[idx[i]] = rng.gen_range(1u8, 4u8);
    }

    let mut b = vec![];

    for i in 0..a.len() {
        match edits[i] {
            0u8 => { // same
                b.push(a[i]);
            },
            1u8 => { // diff
                let mut iter = AMINO_ACIDS.choose_multiple(rng, 2);
                let first = *iter.next().unwrap();
                let second = *iter.next().unwrap();
                b.push(if first == a[i] { second } else { first });
            },
            2u8 => { // insert
                b.push(*AMINO_ACIDS.choose(rng).unwrap());
                b.push(a[i]);
            },
            3u8 => (), // delete
            _ => panic!("This should not have been reached!")
        }
    }

    b
}

fn rand_protein<R: Rng>(length: usize, rng: &mut R) -> Vec<u8> {
    let mut res = vec![0u8; length];

    for i in 0..length {
        res[i] = *AMINO_ACIDS.choose(rng).unwrap();
    }

    res
}
