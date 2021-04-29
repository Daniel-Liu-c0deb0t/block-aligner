#![feature(test)]
#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

extern crate test;
use test::{Bencher, black_box};

use block_aligner::avx2::*;

#[repr(align(32))]
struct A([i16; L]);

#[target_feature(enable = "avx2")]
unsafe fn bench_opt_prefix_scan_core(b: &mut Bencher) {
    let vec = A([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 12, 13, 14, 11]);
    let vec = simd_load(vec.0.as_ptr() as *const Simd);

    b.iter(|| {
        simd_prefix_scan_i16(black_box(vec), -1)
    });
}

#[target_feature(enable = "avx2")]
unsafe fn bench_naive_prefix_scan_core(b: &mut Bencher) {
    let vec = A([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 12, 13, 14, 11]);
    let vec = simd_load(vec.0.as_ptr() as *const Simd);

    b.iter(|| {
        simd_naive_prefix_scan_i16(black_box(vec), -1)
    });
}

#[bench]
fn bench_opt_prefix_scan(b: &mut Bencher) { unsafe { bench_opt_prefix_scan_core(b); } }

#[bench]
fn bench_naive_prefix_scan(b: &mut Bencher) { unsafe { bench_naive_prefix_scan_core(b); } }
