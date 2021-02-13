#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::cmp;

pub type Simd = __m256i;
pub type HalfSimd = __m128i;
pub const L: usize = 16;
pub const L_BYTES: usize = L * 2;

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
pub unsafe fn simd_loadu(ptr: *const Simd) -> Simd { _mm256_loadu_si256(ptr) }

#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn simd_store(ptr: *mut Simd, a: Simd) { _mm256_store_si256(ptr, a) }

#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn simd_storeu(ptr: *mut Simd, a: Simd) { _mm256_storeu_si256(ptr, a) }

#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn simd_set1_i16(v: i16) -> Simd { _mm256_set1_epi16(v) }

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

#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn simd_sl_i16<const NUM: usize>(a: Simd, b: Simd) -> Simd {
    debug_assert!(2 * NUM <= L);
    _mm256_alignr_epi8(a, _mm256_permute2x128_si256(a, b, 0x02), (L - (2 * NUM)) as i32)
}

#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn simd_sr_i16<const NUM: usize>(a: Simd, b: Simd) -> Simd {
    debug_assert!(2 * NUM <= L);
    _mm256_alignr_epi8(_mm256_permute2x128_si256(a, b, 0x03), b, (2 * NUM) as i32)
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn simd_sl_i128(a: Simd, b: Simd) -> Simd {
    _mm256_permute2x128_si256(a, b, 0x02)
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn simd_sl_i192(a: Simd, b: Simd) -> Simd {
    _mm256_alignr_epi8(_mm256_permute2x128_si256(a, b, 0x02), b, 8)
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
pub unsafe fn simd_prefix_scan_i16(delta_R_max: Simd, stride_gap: Simd, neg_inf: Simd) -> Simd {
    let stride_gap2 = _mm256_adds_epi16(stride_gap, stride_gap);
    let stride_gap4 = _mm256_adds_epi16(stride_gap2, stride_gap2);

    // D C B A  D C B A  D C B A  D C B A
    let reduce1 = _mm256_max_epi16(delta_R_max, _mm256_adds_epi16(stride_gap, _mm256_slli_si256(delta_R_max, 2)));
    // D   B    D   B    D   B    D   B
    let mut reduce2 = _mm256_max_epi16(reduce1, _mm256_adds_epi16(stride_gap2, _mm256_slli_si256(reduce1, 4)));
    // D        D        D        D

    // Unrolled loop with multiple accumulators for prefix add and max
    {
        let shift0 = reduce2;
        let mut shift4 = simd_sl_i16::<4>(reduce2, neg_inf);
        let mut shift8 = simd_sl_i128(reduce2, neg_inf);
        let mut shift12 = simd_sl_i192(reduce2, neg_inf);

        let mut stride_gap_sum = stride_gap4;
        shift4 = _mm256_adds_epi16(shift4, stride_gap_sum);
        shift4 = _mm256_max_epi16(shift0, shift4);

        stride_gap_sum = _mm256_adds_epi16(stride_gap_sum, stride_gap4);
        shift8 = _mm256_adds_epi16(shift8, stride_gap_sum);

        stride_gap_sum = _mm256_adds_epi16(stride_gap_sum, stride_gap4);
        shift12 = _mm256_adds_epi16(shift12, stride_gap_sum);
        shift12 = _mm256_max_epi16(shift8, shift12);

        reduce2 = _mm256_max_epi16(shift4, shift12);
    }
    // reduce2
    // D        D        D        D

    let unreduce1_mid = _mm256_max_epi16(_mm256_adds_epi16(simd_sl_i16::<2>(reduce2, neg_inf), stride_gap2), reduce1);
    //     B        B        B        B
    let unreduce1 = _mm256_blend_epi16(reduce2, unreduce1_mid, 0b0010_0010_0010_0010);
    // D   B    D   B    D   B    D   B
    let unreduce2 = _mm256_max_epi16(_mm256_adds_epi16(simd_sl_i16::<1>(unreduce1, neg_inf), stride_gap), delta_R_max);
    //   C   A    C   A    C   A    C   A
    _mm256_blend_epi16(unreduce1, unreduce2, 0b0101_0101_0101_0101)
    // D C B A  D C B A  D C B A  D C B A
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
pub unsafe fn halfsimd_loadu(ptr: *const HalfSimd) -> HalfSimd { _mm_loadu_si128(ptr) }

#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn halfsimd_store(ptr: *mut HalfSimd, a: HalfSimd) { _mm_store_si128(ptr, a) }

#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn halfsimd_storeu(ptr: *mut HalfSimd, a: HalfSimd) { _mm_storeu_si128(ptr, a) }

#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn halfsimd_set1_i8(v: i8) -> HalfSimd { _mm_set1_epi8(v) }

#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn halfsimd_sr_i8<const NUM: usize>(a: HalfSimd, b: HalfSimd) -> HalfSimd {
    debug_assert!(NUM <= L);
    _mm_alignr_epi8(a, b, NUM as i32)
}

#[target_feature(enable = "avx2")]
#[allow(dead_code)]
pub unsafe fn simd_dbg_i16(v: Simd) {
    let mut a = [0i16; L];
    simd_storeu(a.as_mut_ptr() as *mut Simd, v);

    for i in (0..a.len()).rev() {
        print!("{:6} ", a[i]);
    }
    println!();
}

#[target_feature(enable = "avx2")]
#[allow(dead_code)]
pub unsafe fn halfsimd_dbg_i8(v: HalfSimd) {
    let mut a = [0i8; L];
    halfsimd_storeu(a.as_mut_ptr() as *mut HalfSimd, v);

    for i in (0..a.len()).rev() {
        print!("{:3} ", a[i]);
    }
    println!();
}

#[target_feature(enable = "avx2")]
pub unsafe fn simd_assert_vec_eq(a: Simd, b: [i16; L]) {
    let mut arr = [0i16; L];
    simd_storeu(arr.as_mut_ptr() as *mut Simd, a);
    assert_eq!(arr, b);
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
        let vec = [0i16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 12, 13, 14, 11];
        let res = simd_prefix_scan_i16(simd_loadu(vec.as_ptr() as *const Simd), simd_set1_i16(0), simd_set1_i16(i16::MIN));
        simd_assert_vec_eq(res, [0i16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 15, 15, 15, 15]);

        let vec = [0i16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 12, 13, 14, 11];
        let res = simd_prefix_scan_i16(simd_loadu(vec.as_ptr() as *const Simd), simd_set1_i16(-1), simd_set1_i16(i16::MIN));
        simd_assert_vec_eq(res, [0i16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 14, 13, 14, 13]);
    }
}
