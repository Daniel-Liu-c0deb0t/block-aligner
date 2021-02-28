#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::cmp;

pub type Simd = __m256i;
pub type HalfSimd = __m128i;
pub const L: usize = 16;
pub const L_BYTES: usize = L * 2;
pub const HALFSIMD_MUL: usize = 1;

#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn simd_adds_i16(a: Simd, b: Simd) -> Simd { _mm256_adds_epi16(a, b) }

#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn simd_subs_i16(a: Simd, b: Simd) -> Simd { _mm256_subs_epi16(a, b) }

#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn simd_max_i16(a: Simd, b: Simd) -> Simd { _mm256_max_epi16(a, b) }

#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn simd_cmpeq_i16(a: Simd, b: Simd) -> Simd { _mm256_cmpeq_epi16(a, b) }

#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn simd_load(ptr: *const Simd) -> Simd { _mm256_load_si256(ptr) }

#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn simd_store(ptr: *mut Simd, a: Simd) { _mm256_store_si256(ptr, a) }

#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn simd_set1_i16(v: i16) -> Simd { _mm256_set1_epi16(v) }

#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn simd_set4_i16(d: i16, c: i16, b: i16, a: i16) -> Simd { _mm256_set_epi16(d, c, b, a, d, c, b, a, d, c, b, a, d, c, b, a) }

#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn simd_extract_i16<const IDX: usize>(a: Simd) -> i16 {
    debug_assert!(IDX < L);
    _mm256_extract_epi16(a, IDX as i32) as i16
}

#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn simd_insert_i16<const IDX: usize>(a: Simd, v: i16) -> Simd {
    debug_assert!(IDX < L);
    _mm256_insert_epi16(a, v, IDX as i32)
}

#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn simd_movemask_i8(a: Simd) -> u32 { _mm256_movemask_epi8(a) as u32 }

macro_rules! simd_sl_i16 {
    ($a:expr, $b:expr, $num:literal) => {
        {
            debug_assert!(2 * $num <= L);
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            _mm256_alignr_epi8($a, _mm256_permute2x128_si256($a, $b, 0x02), (L - (2 * $num)) as i32)
        }
    };
}

macro_rules! simd_sr_i16 {
    ($a:expr, $b:expr, $num:literal) => {
        {
            debug_assert!(2 * $num <= L);
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            _mm256_alignr_epi8(_mm256_permute2x128_si256($a, $b, 0x03), $b, (2 * $num) as i32)
        }
    };
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn simd_sl_i128(a: Simd, b: Simd) -> Simd {
    _mm256_permute2x128_si256(a, b, 0x02)
}

#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn simd_slow_extract_i16(v: Simd, i: usize) -> i16 {
    debug_assert!(i < L);

    #[repr(align(32))]
    struct A([i16; L]);

    let mut a = A([0i16; L]);
    simd_store(a.0.as_mut_ptr() as *mut Simd, v);
    *a.0.get_unchecked(i)
}

#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn simd_hmax_i16(mut v: Simd) -> i16 {
    v = _mm256_max_epi16(v, _mm256_srli_si256(v, 2));
    v = _mm256_max_epi16(v, _mm256_srli_si256(v, 4));
    v = _mm256_max_epi16(v, _mm256_srli_si256(v, 8));
    cmp::max(simd_extract_i16::<0>(v), simd_extract_i16::<{ L / 2 }>(v))
}

#[target_feature(enable = "avx2")]
#[inline]
#[allow(non_snake_case)]
pub unsafe fn simd_prefix_scan_i16(delta_R_max: Simd, stride_gap: Simd, stride_gap1234: Simd, neg_inf: Simd) -> Simd {
    // Optimized prefix add and max for every four elements
    let mut shift1 = simd_sl_i16!(delta_R_max, neg_inf, 1);
    shift1 = _mm256_adds_epi16(shift1, stride_gap);
    shift1 = _mm256_max_epi16(shift1, delta_R_max);
    let mut shift2 = simd_sl_i16!(shift1, neg_inf, 2);
    shift2 = _mm256_adds_epi16(shift2, _mm256_slli_epi16(stride_gap, 1));
    shift2 = _mm256_max_epi16(shift1, shift2);

    // Optimized prefix add and max for every group of four elements
    let mut shift4 = simd_sl_i16!(shift2, neg_inf, 4);
    shift4 = _mm256_adds_epi16(shift4, _mm256_slli_epi16(stride_gap, 2));
    shift4 = _mm256_max_epi16(shift2, shift4);
    let mut shift8 = simd_sl_i128(shift4, neg_inf);
    shift8 = _mm256_adds_epi16(shift8, _mm256_slli_epi16(stride_gap, 3));
    let temp = _mm256_max_epi16(shift4, shift8);

    // Almost there: correct each group using an element from the previous group
    let mut correct = simd_sl_i16!(temp, neg_inf, 1);

    correct = _mm256_shufflelo_epi16(correct, 0);
    correct = _mm256_shufflehi_epi16(correct, 0);
    correct = _mm256_adds_epi16(correct, stride_gap1234);

    _mm256_max_epi16(temp, correct)
}

// use avx2 target feature to prevent legacy SSE mode penalty

#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn halfsimd_lookup2_i16(lut1: HalfSimd, lut2: HalfSimd, v: HalfSimd) -> Simd {
    let a = _mm_shuffle_epi8(lut1, v);
    let b = _mm_shuffle_epi8(lut2, v);
    let mask = _mm_cmpgt_epi8(_mm_set1_epi8(0b00010000), v);
    let c = _mm_blendv_epi8(b, a, mask);
    _mm256_cvtepi8_epi16(c)
}

#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn halfsimd_lookup1_i16(lut: HalfSimd, v: HalfSimd) -> Simd {
    _mm256_cvtepi8_epi16(_mm_shuffle_epi8(lut, v))
}

#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn halfsimd_load(ptr: *const HalfSimd) -> HalfSimd { _mm_load_si128(ptr) }

#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn halfsimd_store(ptr: *mut HalfSimd, a: HalfSimd) { _mm_store_si128(ptr, a) }

#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn halfsimd_set1_i8(v: i8) -> HalfSimd { _mm_set1_epi8(v) }

#[inline]
pub fn halfsimd_get_idx(i: usize) -> usize { i }

macro_rules! halfsimd_sr_i8 {
    ($a:expr, $b:expr, $num:literal) => {
        {
            debug_assert!($num <= L);
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            _mm_alignr_epi8($a, $b, $num as i32)
        }
    };
}

#[target_feature(enable = "avx2")]
#[allow(dead_code)]
pub unsafe fn simd_dbg_i16(v: Simd) {
    #[repr(align(32))]
    struct A([i16; L]);

    let mut a = A([0i16; L]);
    simd_store(a.0.as_mut_ptr() as *mut Simd, v);

    for i in (0..a.0.len()).rev() {
        print!("{:6} ", a.0[i]);
    }
    println!();
}

#[target_feature(enable = "avx2")]
#[allow(dead_code)]
pub unsafe fn halfsimd_dbg_i8(v: HalfSimd) {
    #[repr(align(16))]
    struct A([i8; L]);

    let mut a = A([0i8; L]);
    halfsimd_store(a.0.as_mut_ptr() as *mut HalfSimd, v);

    for i in (0..a.0.len()).rev() {
        print!("{:3} ", a.0[i]);
    }
    println!();
}

#[target_feature(enable = "avx2")]
pub unsafe fn simd_assert_vec_eq(a: Simd, b: [i16; L]) {
    #[repr(align(32))]
    struct A([i16; L]);

    let mut arr = A([0i16; L]);
    simd_store(arr.0.as_mut_ptr() as *mut Simd, a);
    assert_eq!(arr.0, b);
}

#[target_feature(enable = "avx2")]
pub unsafe fn halfsimd_assert_vec_eq(a: HalfSimd, b: [i8; L]) {
    #[repr(align(32))]
    struct A([i8; L]);

    let mut arr = A([0i8; L]);
    halfsimd_store(arr.0.as_mut_ptr() as *mut HalfSimd, a);
    assert_eq!(arr.0, b);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_scan() {
        unsafe { test_prefix_scan_core() };
    }

    #[target_feature(enable = "avx2")]
    unsafe fn test_prefix_scan_core() {
        #[repr(align(32))]
        struct A([i16; L]);

        let vec = A([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 12, 13, 14, 11]);
        let res = simd_prefix_scan_i16(simd_load(vec.0.as_ptr() as *const Simd), simd_set1_i16(0), simd_set4_i16(0, 0, 0, 0), simd_set1_i16(i16::MIN));
        simd_assert_vec_eq(res, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 15, 15, 15, 15]);

        let vec = A([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 12, 13, 14, 11]);
        let res = simd_prefix_scan_i16(simd_load(vec.0.as_ptr() as *const Simd), simd_set1_i16(-1), simd_set4_i16(-4, -3, -2, -1), simd_set1_i16(i16::MIN));
        simd_assert_vec_eq(res, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 14, 13, 14, 13]);
    }
}
