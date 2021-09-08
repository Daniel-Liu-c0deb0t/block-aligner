//! SIMD-accelerated library for computing global and X-drop affine
//! gap sequence alignments using an adaptive block-based algorithm.
//!
//! Currently, AVX2 and WASM SIMD are supported.
//!
//! ## Example
//! ```
//! use block_aligner::scan_block::*;
//! use block_aligner::scores::*;
//! use block_aligner::cigar::*;
//!
//! let block_size = 16;
//! let gaps = Gaps { open: -2, extend: -1 };
//! let r = PaddedBytes::from_bytes::<NucMatrix>(b"TTAAAAAAATTTTTTTTTTTT", block_size);
//! let q = PaddedBytes::from_bytes::<NucMatrix>(b"TTTTTTTTAAAAAAATTTTTTTTT", block_size);
//!
//! // Align with traceback, but no x drop threshold.
//! let a = Block::<_, true, false>::align(&q, &r, &NW1, gaps, block_size..=block_size, 0);
//! let res = a.res();
//!
//! assert_eq!(res, AlignResult { score: 7, query_idx: 24, reference_idx: 21 });
//! assert_eq!(a.trace().cigar(res.query_idx, res.reference_idx).to_string(), "2M6I16M3D");
//! ```
//!
//! When building your code that uses this library, it is important to specify the
//! correct flags to turn on specific target features that this library uses.
//!
//! For x86 AVX2:
//! ```text
//! RUSTFLAGS="-C target-cpu=native" cargo build --release
//! ```
//! or
//! ```text
//! RUSTFLAGS="-C target-feature=+avx2" cargo build --release
//! ```
//!
//! For WASM SIMD:
//! ```text
//! RUSTFLAGS="-C target-feature=+simd128" cargo build --target=wasm32-wasi --release
//! ```

#![cfg_attr(feature = "mca", feature(asm))]

//use wee_alloc::WeeAlloc;

//#[global_allocator]
//static ALLOC: WeeAlloc = WeeAlloc::INIT;

// special SIMD instruction set modules adapted for this library
// their types and lengths are abstracted out

#[cfg(feature = "simd_avx2")]
#[macro_use]
#[doc(hidden)]
/// cbindgen:ignore
pub mod avx2;

#[cfg(feature = "simd_avx2")]
pub use avx2::L;

#[cfg(feature = "simd_wasm")]
#[macro_use]
#[doc(hidden)]
/// cbindgen:ignore
pub mod simd128;

#[cfg(feature = "simd_wasm")]
pub use simd128::L;

#[cfg(any(feature = "simd_avx2", feature = "simd_wasm"))]
pub mod scan_block;
#[cfg(any(feature = "simd_avx2", feature = "simd_wasm"))]
pub mod scores;
#[cfg(any(feature = "simd_avx2", feature = "simd_wasm"))]
pub mod cigar;
#[cfg(any(feature = "simd_avx2", feature = "simd_wasm"))]
pub mod simulate;

#[cfg(feature = "simd_avx2")]
#[doc(hidden)]
pub mod ffi;
