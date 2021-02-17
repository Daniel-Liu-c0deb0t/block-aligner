use std::arch::wasm32::*;

use std::cmp;

pub type Simd = v128;
pub type HalfSimd = v128;
pub const L: usize = 8;
pub const L_BYTES: usize = L * 2;

#[target_feature(enable = "simd128")]
#[inline]
pub unsafe fn simd_adds_i16(a: Simd, b: Simd) -> Simd { i16x8_add_saturate_s(a, b) }

#[target_feature(enable = "simd128")]
#[inline]
pub unsafe fn simd_subs_i16(a: Simd, b: Simd) -> Simd { i16x8_sub_saturate_s(a, b) }

#[target_feature(enable = "simd128")]
#[inline]
pub unsafe fn simd_max_i16(a: Simd, b: Simd) -> Simd { i16x8_max_s(a, b) }

#[target_feature(enable = "simd128")]
#[inline]
pub unsafe fn simd_cmpeq_i16(a: Simd, b: Simd) -> Simd { i16x8_eq(a, b) }

#[target_feature(enable = "simd128")]
#[inline]
pub unsafe fn simd_load(ptr: *const Simd) -> Simd { v128_load(ptr) }

#[target_feature(enable = "simd128")]
#[inline]
pub unsafe fn simd_store(ptr: *mut Simd, a: Simd) { v128_store(ptr, a) }

#[target_feature(enable = "simd128")]
#[inline]
pub unsafe fn simd_set1_i16(v: i16) -> Simd { i16x8_splat(v) }

#[target_feature(enable = "simd128")]
#[inline]
pub unsafe fn simd_extract_i16<const IDX: usize>(a: Simd) -> i16 {
    debug_assert!(IDX < L);
    i16x8_extract_lane::<{ IDX }>(a)
}

#[target_feature(enable = "simd128")]
#[inline]
pub unsafe fn simd_insert_i16<const IDX: usize>(a: Simd, v: i16) -> Simd {
    debug_assert!(IDX < L);
    i16x8_replace_lane::<{ IDX }>(a, v)
}

#[target_feature(enable = "simd128")]
#[inline]
pub unsafe fn simd_movemask_i8(a: Simd) -> u32 {
    //i8x16_bitmask(a) as u32
    const mul: u64 = {
        let mul = 0u64;
        mul |= 1u64 << (0 - 0);
        mul |= 1u64 << (8 - 1);
        mul |= 1u64 << (16 - 2);
        mul |= 1u64 << (24 - 3);
        mul |= 1u64 << (32 - 4);
        mul |= 1u64 << (40 - 5);
        mul |= 1u64 << (48 - 6);
        mul |= 1u64 << (56 - 7);
        mul
    };
    let b = i64x2_mul(i8x16_and(a, i8x16_splat(0b10000000u8 as i8)), i64x2_splat(mul));
    let res1 = i8x16_extract_lane::<{ L * 2 - 1 }>(b) as u32;
    let res2 = i8x16_extract_lane::<{ L - 1 }>(b) as u32;
    (res1 << 8) | res2
}

macro_rules! simd_sl_i16 {
    ($a:expr, $b:expr, $num:literal) => {
        {
            debug_assert!($num <= L);
            use std::arch::wasm32::*;
            v16x8_shuffle::<{ 0 + $num }, { 1 + $num }, { 2 + $num }, { 3 + $num }, { 4 + $num }, { 5 + $num }, { 6 + $num }, { 7 + $num }>($a, $b)
        }
    };
}

macro_rules! simd_sr_i16 {
    ($a:expr, $b:expr, $num:literal) => {
        {
            debug_assert!($num <= L);
            use std::arch::wasm32::*;
            v16x8_shuffle::<{ 8 - $num }, { 9 - $num }, { 10 - $num }, { 11 - $num }, { 12 - $num }, { 13 - $num }, { 14 - $num }, { 15 - $num }>($a, $b)
        }
    };
}

#[target_feature(enable = "simd128")]
#[inline]
pub unsafe fn simd_slow_extract_i16(v: Simd, i: usize) -> i16 {
    debug_assert!(i < L);

    #[repr(align(16))]
    struct A([i16; L]);

    let mut a = A([0i16; L]);
    simd_store(a.0.as_mut_ptr() as *mut Simd, v);
    *a.0.get_unchecked(i)
}

#[target_feature(enable = "simd128")]
#[inline]
pub unsafe fn simd_hmax_i16(mut v: Simd) -> i16 {
    v = i16x8_max_s(v, simd_sr_i16!(v, v, 1));
    v = i16x8_max_s(v, simd_sr_i16!(v, v, 2));
    v = i16x8_max_s(v, simd_sr_i16!(v, v, 4));
    cmp::max(simd_extract_i16::<0>(v), simd_extract_i16::<{ L / 2 }>(v))
}

#[target_feature(enable = "simd128")]
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

#[target_feature(enable = "simd128")]
#[inline]
pub unsafe fn halfsimd_lookup2_i16(lut1: HalfSimd, lut2: HalfSimd, v: HalfSimd) -> Simd {
    let mask = i8x16_splat(0b1111);
    let v = v128_and(v, mask);
    let a = v8x16_swizzle(lut1, v);
    let b = v8x16_swizzle(lut2, v);
    let mask = i8x16_gt_s(v, mask);
    let c = v128_bitselect(b, a, mask);
    i16x8_widen_low_i8x16_s(c)
}

#[target_feature(enable = "simd128")]
#[inline]
pub unsafe fn halfsimd_lookup1_i16(lut: HalfSimd, v: HalfSimd) -> Simd {
    i16x8_widen_low_i8x16_s(v8x16_swizzle(lut, v128_and(v, i8x16_splat(0b1111))))
}

#[target_feature(enable = "simd128")]
#[inline]
pub unsafe fn halfsimd_load(ptr: *const HalfSimd) -> HalfSimd { v128_load(ptr) }

#[target_feature(enable = "simd128")]
#[inline]
pub unsafe fn halfsimd_store(ptr: *mut HalfSimd, a: HalfSimd) { v128_store(ptr, a) }

#[target_feature(enable = "simd128")]
#[inline]
pub unsafe fn halfsimd_set1_i8(v: i8) -> HalfSimd { i16x8_splat(v) }

macro_rules! halfsimd_sr_i8 {
    ($a:expr, $b:expr, $num:literal) => {
        {
            debug_assert!($num <= L);
            use std::arch::wasm32::*;
            v8x16_shuffle::<{ 16 - $num }, { 17 - $num }, { 18 - $num }, { 19 - $num }, { 20 - $num }, { 21 - $num }, { 22 - $num }, { 23 - $num },
                { 24 - $num }, { 25 - $num }, { 26 - $num }, { 27 - $num }, { 28 - $num }, { 29 - $num }, { 30 - $num }, { 31 - $num }>($a, $b)
        }
    };
}

#[target_feature(enable = "simd128")]
#[allow(dead_code)]
pub unsafe fn simd_dbg_i16(v: Simd) {
    #[repr(align(16))]
    struct A([i16; L]);

    let mut a = A([0i16; L]);
    simd_store(a.0.as_mut_ptr() as *mut Simd, v);

    for i in (0..a.0.len()).rev() {
        print!("{:6} ", a.0[i]);
    }
    println!();
}

#[target_feature(enable = "simd128")]
#[allow(dead_code)]
pub unsafe fn halfsimd_dbg_i8(v: HalfSimd) {
    #[repr(align(16))]
    struct A([i8; L]);

    let mut a = A([0i8; L]);
    halfsimd_store(a.0.as_mut_ptr() as *mut HalfSimd, v);

    for i in (0..a.0.len() / 2).rev() {
        print!("{:3} ", a.0[i]);
    }
    println!();
}

#[target_feature(enable = "simd128")]
pub unsafe fn simd_assert_vec_eq(a: Simd, b: [i16; L]) {
    #[repr(align(16))]
    struct A([i16; L]);

    let mut arr = A([0i16; L]);
    simd_store(arr.0.as_mut_ptr() as *mut Simd, a);
    assert_eq!(arr.0, b);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_scan() {
        unsafe { test_prefix_scan_core() };
    }

    #[target_feature(enable = "simd128")]
    unsafe fn test_prefix_scan_core() {
        #[repr(align(16))]
        struct A([i16; L]);

        let vec = A([8, 9, 10, 15, 12, 13, 14, 11]);
        let res = simd_prefix_scan_i16(simd_load(vec.0.as_ptr() as *const Simd), simd_set1_i16(0), simd_set1_i16(i16::MIN));
        simd_assert_vec_eq(res, [8, 9, 10, 15, 15, 15, 15, 15]);

        let vec = A([8, 9, 10, 15, 12, 13, 14, 11]);
        let res = simd_prefix_scan_i16(simd_load(vec.0.as_ptr() as *const Simd), simd_set1_i16(-1), simd_set1_i16(i16::MIN));
        simd_assert_vec_eq(res, [8, 9, 10, 15, 14, 13, 14, 13]);
    }
}
