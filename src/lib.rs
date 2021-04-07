#![cfg_attr(target_arch = "wasm32", feature(wasm_simd))]
#![cfg_attr(target_arch = "wasm32", feature(wasm_target_feature))]

use wee_alloc::WeeAlloc;

#[global_allocator]
static ALLOC: WeeAlloc = WeeAlloc::INIT;

// special SIMD instruction set modules adapted for this library
// their types and lengths are abstracted out

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[macro_use]
pub mod avx2;

#[cfg(target_arch = "wasm32")]
#[macro_use]
pub mod simd128;

#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "wasm32"))]
pub mod scan_minecraft;

pub mod scores;
