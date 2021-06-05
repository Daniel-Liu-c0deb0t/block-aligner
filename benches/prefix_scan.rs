#![feature(test)]
#![cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"))]

extern crate test;
use test::{Bencher, black_box};

use block_aligner::avx2::*;

#[repr(align(32))]
struct A([i16; L]);

#[bench]
fn bench_opt_prefix_scan(b: &mut Bencher) {
    unsafe {
        let vec = A([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 12, 13, 14, 11]);
        let vec = simd_load(vec.0.as_ptr() as *const Simd);

        b.iter(|| {
            let consts = get_prefix_scan_consts(-1);
            simd_prefix_scan_i16(black_box(vec), consts)
        });
    }
}

#[bench]
fn bench_naive_prefix_scan(b: &mut Bencher) {
    unsafe {
        let vec = A([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 12, 13, 14, 11]);
        let vec = simd_load(vec.0.as_ptr() as *const Simd);

        b.iter(|| {
            let consts = get_prefix_scan_consts(-1);
            simd_naive_prefix_scan_i16(black_box(vec), consts)
        });
    }
}
