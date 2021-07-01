#![feature(core_intrinsics)]
#![feature(asm)]
#![feature(vec_into_raw_parts)]

//use wee_alloc::WeeAlloc;

//#[global_allocator]
//static ALLOC: WeeAlloc = WeeAlloc::INIT;

// special SIMD instruction set modules adapted for this library
// their types and lengths are abstracted out

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"))]
#[macro_use]
#[doc(hidden)]
/// cbindgen:ignore
pub mod avx2;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[macro_use]
#[doc(hidden)]
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
#[doc(hidden)]
pub mod ffi;
