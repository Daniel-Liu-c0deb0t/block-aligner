#![cfg_attr(target_arch = "wasm32", feature(wasm_simd))]
#![feature(core_intrinsics)]
#![feature(asm)]

//use wee_alloc::WeeAlloc;

//#[global_allocator]
//static ALLOC: WeeAlloc = WeeAlloc::INIT;

// special SIMD instruction set modules adapted for this library
// their types and lengths are abstracted out

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"))]
#[macro_use]
/// cbindgen:ignore
pub mod avx2;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[macro_use]
/// cbindgen:ignore
pub mod simd128;

#[cfg(any(
        all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"),
        all(target_arch = "wasm32", target_feature = "simd128")
))]
pub mod scan_block;

#[cfg(any(
        all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"),
        all(target_arch = "wasm32", target_feature = "simd128")
))]
pub mod scores;

pub mod cigar;
pub mod simulate;

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"))]
pub mod ffi;
