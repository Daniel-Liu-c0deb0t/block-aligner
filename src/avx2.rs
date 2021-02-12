#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#![cfg(target_feature = "avx2")]

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub type Simd = __m256i;
pub const L: usize = 16;
pub const L_BYTES: usize = L * 2;

#[target_feature(enable = "avx2")]
#[inline]
pub fn simd_adds_i16(a: Simd, b: Simd) -> Simd { _mm256_adds_epi16(a, b) }

#[target_feature(enable = "avx2")]
#[inline]
pub fn simd_subs_i16(a: Simd, b: Simd) -> Simd { _mm256_subs_epi16(a, b) }

#[target_feature(enable = "avx2")]
#[inline]
pub fn simd_max_i16(a: Simd, b: Simd) -> Simd { _mm256_max_epi16(a, b) }

#[target_feature(enable = "avx2")]
#[inline]
pub fn simd_cmpeq_i16(a: Simd, b: Simd) -> Simd { _mm256_cmpeq_epi16(a, b) }

#[target_feature(enable = "avx2")]
#[inline]
pub fn simd_load(ptr: *const Simd) -> Simd { _mm256_load_si256(ptr) }

#[target_feature(enable = "avx2")]
#[inline]
pub fn simd_loadu(ptr: *const Simd) -> Simd { _mm256_loadu_si256(ptr) }

#[target_feature(enable = "avx2")]
#[inline]
pub fn simd_store(ptr: *mut Simd, a: Simd) { _mm256_store_si256(ptr, a) }

#[target_feature(enable = "avx2")]
#[inline]
pub fn simd_storeu(ptr: *mut Simd, a: Simd) { _mm256_storeu_si256(ptr, a) }

#[target_feature(enable = "avx2")]
#[inline]
pub fn simd_set1_i16(v: i16) -> Simd { _mm256_set1_epi16(v) }

#[target_feature(enable = "avx2")]
#[inline]
pub fn simd_extract_i16<const IDX: usize>(a: Simd) -> i16 { _mm256_extract_epi16(a, IDX as i32) as i16 }

#[target_feature(enable = "avx2")]
#[inline]
pub fn simd_insert_i16<const IDX: usize>(a: Simd, v: i16) -> Simd { _mm256_insert_epi16(a, v, IDX as i32) }

#[target_feature(enable = "avx2")]
#[inline]
pub fn simd_movemask_i8(a: Simd) -> u32 { _mm256_movemask_epi8(a) as u32 }
