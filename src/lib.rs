#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[macro_use]
pub mod avx2;

#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "wasm32"))]
pub mod scan_avx2;

pub mod scores;
