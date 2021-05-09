#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub type Simd = __m256i;
pub type HalfSimd = __m128i;
pub type TraceType = u32;
pub const L: usize = 16;
pub const L_BYTES: usize = L * 2;
pub const HALFSIMD_MUL: usize = 1;

#[inline]
pub unsafe fn simd_adds_i16(a: Simd, b: Simd) -> Simd { _mm256_adds_epi16(a, b) }

#[inline]
pub unsafe fn simd_subs_i16(a: Simd, b: Simd) -> Simd { _mm256_subs_epi16(a, b) }

#[inline]
pub unsafe fn simd_max_i16(a: Simd, b: Simd) -> Simd { _mm256_max_epi16(a, b) }

#[inline]
pub unsafe fn simd_cmpeq_i16(a: Simd, b: Simd) -> Simd { _mm256_cmpeq_epi16(a, b) }

#[inline]
pub unsafe fn simd_cmpgt_i16(a: Simd, b: Simd) -> Simd { _mm256_cmpgt_epi16(a, b) }

#[inline]
pub unsafe fn simd_blend_i8(a: Simd, b: Simd, mask: Simd) -> Simd { _mm256_blendv_epi8(a, b, mask) }

#[inline]
pub unsafe fn simd_load(ptr: *const Simd) -> Simd { _mm256_load_si256(ptr) }

#[inline]
pub unsafe fn simd_store(ptr: *mut Simd, a: Simd) { _mm256_store_si256(ptr, a) }

#[inline]
pub unsafe fn simd_set1_i16(v: i16) -> Simd { _mm256_set1_epi16(v) }

#[macro_export]
macro_rules! simd_extract_i16 {
    ($a:expr, $num:expr) => {
        {
            debug_assert!($num < L);
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            _mm256_extract_epi16($a, $num as i32) as i16
        }
    };
}

#[macro_export]
macro_rules! simd_insert_i16 {
    ($a:expr, $v:expr, $num:expr) => {
        {
            debug_assert!($num < L);
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            _mm256_insert_epi16($a, $v, $num as i32)
        }
    };
}

#[inline]
pub unsafe fn simd_movemask_i8(a: Simd) -> u32 { _mm256_movemask_epi8(a) as u32 }

#[macro_export]
macro_rules! simd_sl_i16 {
    ($a:expr, $b:expr, $num:expr) => {
        {
            debug_assert!(2 * $num <= L);
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            if $num == L / 2 {
                _mm256_permute2x128_si256($a, $b, 0x03)
            } else {
                _mm256_alignr_epi8($a, _mm256_permute2x128_si256($a, $b, 0x03), (L - (2 * $num)) as i32)
            }
        }
    };
}

#[macro_export]
macro_rules! simd_sr_i16 {
    ($a:expr, $b:expr, $num:expr) => {
        {
            debug_assert!(2 * $num <= L);
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            if $num == L / 2 {
                _mm256_permute2x128_si256($a, $b, 0x03)
            } else {
                _mm256_alignr_epi8(_mm256_permute2x128_si256($a, $b, 0x03), $b, (2 * $num) as i32)
            }
        }
    };
}

#[inline]
unsafe fn simd_sl_i128(a: Simd, b: Simd) -> Simd {
    _mm256_permute2x128_si256(a, b, 0x03)
}


#[macro_export]
macro_rules! simd_sll_i16 {
    ($a:expr, $b:expr, $num:expr) => {
        {
            debug_assert!(2 * $num < L);
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            _mm256_alignr_epi8($a, $b, (L - $num * 2) as i32)
        }
    };
}

#[inline]
pub unsafe fn simd_slow_extract_i16(v: Simd, i: usize) -> i16 {
    debug_assert!(i < L);

    #[repr(align(32))]
    struct A([i16; L]);

    let mut a = A([0i16; L]);
    simd_store(a.0.as_mut_ptr() as *mut Simd, v);
    *a.0.get_unchecked(i)
}

#[inline]
pub unsafe fn simd_hmax_i16(v: Simd) -> i16 {
    let mut v2 = _mm256_max_epi16(v, _mm256_srli_si256(v, 2));
    v2 = _mm256_max_epi16(v2, _mm256_srli_si256(v2, 4));
    v2 = _mm256_max_epi16(v2, _mm256_srli_si256(v2, 8));
    v2 = _mm256_max_epi16(v2, simd_sl_i128(v2, v2));
    simd_extract_i16!(v2, 0)
}

#[inline]
pub unsafe fn simd_hargmax_i16(v: Simd, max: i16) -> usize {
    let v2 = _mm256_cmpeq_epi16(v, _mm256_set1_epi16(max));
    (simd_movemask_i8(v2).trailing_zeros() as usize) / 2
}

#[inline]
#[allow(non_snake_case)]
#[allow(dead_code)]
pub unsafe fn simd_naive_prefix_scan_i16(R_max: Simd, gap: i16) -> Simd {
    let (gap_cost, _gap_cost12345678, neg_inf) = get_prefix_scan_consts(gap);
    let mut curr = R_max;

    for _i in 0..(L - 1) {
        let prev = curr;
        curr = simd_sl_i16!(curr, neg_inf, 1);
        curr = _mm256_adds_epi16(curr, gap_cost);
        curr = _mm256_max_epi16(curr, prev);
    }

    curr
}

#[inline]
unsafe fn get_prefix_scan_consts(gap: i16) -> (Simd, Simd, Simd) {
    let gap_cost = _mm256_set1_epi16(gap);
    let gap_cost12345678 = _mm256_set_epi16(
        gap * 8, gap * 7, gap * 6, gap * 5,
        gap * 4, gap * 3, gap * 2, gap * 1,
        gap * 8, gap * 7, gap * 6, gap * 5,
        gap * 4, gap * 3, gap * 2, gap * 1
    );
    let neg_inf = _mm256_set1_epi16(i16::MIN);
    (gap_cost, gap_cost12345678, neg_inf)
}

#[inline]
#[allow(non_snake_case)]
pub unsafe fn simd_prefix_scan_i16(R_max: Simd, gap: i16) -> Simd {
    let (gap_cost, gap_cost12345678, neg_inf) = get_prefix_scan_consts(gap);

    // Optimized prefix add and max for every eight elements
    // Note: be very careful to avoid lane-crossing which has a large penalty
    let mut shift1 = simd_sll_i16!(R_max, neg_inf, 1);
    shift1 = _mm256_adds_epi16(shift1, gap_cost);
    shift1 = _mm256_max_epi16(R_max, shift1);
    let mut shift2 = simd_sll_i16!(shift1, neg_inf, 2);
    shift2 = _mm256_adds_epi16(shift2, _mm256_slli_epi16(gap_cost, 1));
    shift2 = _mm256_max_epi16(shift1, shift2);
    let mut shift4 = simd_sll_i16!(shift2, neg_inf, 4);
    shift4 = _mm256_adds_epi16(shift4, _mm256_slli_epi16(gap_cost, 2));
    shift4 = _mm256_max_epi16(shift2, shift4);

    // Correct the upper lane using the last element of the lower lane
    let mut correct1 = simd_sl_i128(shift4, neg_inf);
    let mask = _mm256_set_epi8(
        15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14,
         1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0
    );
    correct1 = _mm256_shuffle_epi8(correct1, mask);
    correct1 = _mm256_adds_epi16(correct1, gap_cost12345678);
    _mm256_max_epi16(shift4, correct1)
}

#[inline]
pub unsafe fn halfsimd_lookup2_i16(lut1: HalfSimd, lut2: HalfSimd, v: HalfSimd) -> Simd {
    let a = _mm_shuffle_epi8(lut1, v);
    let b = _mm_shuffle_epi8(lut2, v);
    let mask = _mm_cmpgt_epi8(_mm_set1_epi8(0b00010000), v);
    let c = _mm_blendv_epi8(b, a, mask);
    _mm256_cvtepi8_epi16(c)
}

#[inline]
pub unsafe fn halfsimd_lookup1_i16(lut: HalfSimd, v: HalfSimd) -> Simd {
    _mm256_cvtepi8_epi16(_mm_shuffle_epi8(lut, v))
}

#[inline]
pub unsafe fn halfsimd_load(ptr: *const HalfSimd) -> HalfSimd { _mm_load_si128(ptr) }

#[inline]
pub unsafe fn halfsimd_loadu(ptr: *const HalfSimd) -> HalfSimd { _mm_loadu_si128(ptr) }

#[inline]
pub unsafe fn halfsimd_store(ptr: *mut HalfSimd, a: HalfSimd) { _mm_store_si128(ptr, a) }

#[inline]
pub unsafe fn halfsimd_sub_i8(a: HalfSimd, b: HalfSimd) -> HalfSimd { _mm_sub_epi8(a, b) }

#[inline]
pub unsafe fn halfsimd_set1_i8(v: i8) -> HalfSimd { _mm_set1_epi8(v) }

#[inline]
pub unsafe fn halfsimd_get_idx(i: usize) -> usize { i }

#[macro_export]
macro_rules! halfsimd_sr_i8 {
    ($a:expr, $b:expr, $num:expr) => {
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

#[allow(dead_code)]
pub unsafe fn simd_assert_vec_eq(a: Simd, b: [i16; L]) {
    #[repr(align(32))]
    struct A([i16; L]);

    let mut arr = A([0i16; L]);
    simd_store(arr.0.as_mut_ptr() as *mut Simd, a);
    assert_eq!(arr.0, b);
}

#[allow(dead_code)]
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
        unsafe {
            #[repr(align(32))]
            struct A([i16; L]);

            let vec = A([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 12, 13, 14, 11]);
            let res = simd_prefix_scan_i16(simd_load(vec.0.as_ptr() as *const Simd), 0);
            simd_assert_vec_eq(res, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 15, 15, 15, 15]);

            let vec = A([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 12, 13, 14, 11]);
            let res = simd_prefix_scan_i16(simd_load(vec.0.as_ptr() as *const Simd), -1);
            simd_assert_vec_eq(res, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 14, 13, 14, 13]);
        }
    }
}
