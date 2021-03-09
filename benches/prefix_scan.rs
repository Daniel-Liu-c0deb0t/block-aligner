#![feature(test)]
#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

extern crate test;
use test::{Bencher, black_box};

use better_alignment::avx2::*;

#[repr(align(32))]
struct A([i16; L]);

#[target_feature(enable = "avx2")]
unsafe fn bench_opt_prefix_scan_core(b: &mut Bencher) {
    let vec = A([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 12, 13, 14, 11]);
    let vec = simd_load(vec.0.as_ptr() as *const Simd);
    let stride_gap = simd_set1_i16(-1);
    let stride_gap1234 = simd_set4_i16(-4, -3, -2, -1);
    let neg_inf = simd_set1_i16(i16::MIN);

    b.iter(|| {
        simd_prefix_scan_i16(black_box(vec), black_box(stride_gap), black_box(stride_gap1234), neg_inf)
    });
}

#[target_feature(enable = "avx2")]
unsafe fn bench_naive_prefix_scan_core(b: &mut Bencher) {
    let vec = A([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 12, 13, 14, 11]);
    let vec = simd_load(vec.0.as_ptr() as *const Simd);
    let stride_gap = simd_set1_i16(-1);
    let neg_inf = simd_set1_i16(i16::MIN);

    b.iter(|| {
        simd_naive_prefix_scan_i16(black_box(vec), black_box(stride_gap), neg_inf)
    });
}

#[bench]
fn bench_opt_prefix_scan(b: &mut Bencher) { unsafe { bench_opt_prefix_scan_core(b); } }

#[bench]
fn bench_naive_prefix_scan(b: &mut Bencher) { unsafe { bench_naive_prefix_scan_core(b); } }
