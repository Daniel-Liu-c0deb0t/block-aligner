use std::arch::wasm32::*;

pub type Simd = v128;
// no v64 type, so HalfSimd is just v128 with upper half ignored
pub type HalfSimd = v128;
pub const HALFSIMD_MUL: usize = 2;
pub const L: usize = 8;
pub const L_BYTES: usize = L * 2;

// Note: SIMD vectors treated as little-endian

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
pub unsafe fn simd_cmpgt_i16(a: Simd, b: Simd) -> Simd { i16x8_gt(a, b) }

#[target_feature(enable = "simd128")]
#[inline]
pub unsafe fn simd_blend_i8(a: Simd, b: Simd, mask: Simd) -> Simd { v128_bitselect(b, a, mask) }

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
    const MUL: i64 = {
        let mut m = 0u64;
        m |= 1u64 << (0 - 0);
        m |= 1u64 << (8 - 1);
        m |= 1u64 << (16 - 2);
        m |= 1u64 << (24 - 3);
        m |= 1u64 << (32 - 4);
        m |= 1u64 << (40 - 5);
        m |= 1u64 << (48 - 6);
        m |= 1u64 << (56 - 7);
        m as i64
    };
    let b = i64x2_mul(v128_and(a, i8x16_splat(0b10000000u8 as i8)), i64x2_splat(MUL));
    let res1 = i8x16_extract_lane::<{ L * 2 - 1 }>(b) as u32;
    let res2 = i8x16_extract_lane::<{ L - 1 }>(b) as u32;
    (res1 << 8) | res2
}

macro_rules! simd_sl_i16 {
    ($a:expr, $b:expr, $num:literal) => {
        {
            debug_assert!($num <= L);
            use std::arch::wasm32::*;
            v16x8_shuffle::<{ 8 - $num }, { 9 - $num }, { 10 - $num }, { 11 - $num }, { 12 - $num }, { 13 - $num }, { 14 - $num }, { 15 - $num }>($b, $a)
        }
    };
}

#[allow(unused_macros)]
macro_rules! simd_sr_i16 {
    ($a:expr, $b:expr, $num:literal) => {
        {
            debug_assert!($num <= L);
            use std::arch::wasm32::*;
            v16x8_shuffle::<{ 0 + $num }, { 1 + $num }, { 2 + $num }, { 3 + $num }, { 4 + $num }, { 5 + $num }, { 6 + $num }, { 7 + $num }>($b, $a)
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
pub unsafe fn simd_hmax_i16(v: Simd) -> i16 {
    let mut v2 = i16x8_max_s(v, simd_sr_i16!(v, v, 1));
    v2 = i16x8_max_s(v2, simd_sr_i16!(v2, v2, 2));
    v2 = i16x8_max_s(v2, simd_sr_i16!(v2, v2, 4));
    simd_extract_i16::<0>(v2)
}

#[target_feature(enable = "simd128")]
#[inline]
pub unsafe fn simd_hargmax_i16(v: Simd) -> (i16, usize) {
    let max = simd_hmax_i16(v);
    let v2 = i16x8_eq(v, i16x8_splat(max));
    let max_idx = (simd_movemask_i8(v2).trailing_zeros() as usize) / 2;
    (max, max_idx)
}

#[target_feature(enable = "simd128")]
#[inline]
#[allow(non_snake_case)]
#[allow(dead_code)]
pub unsafe fn simd_naive_prefix_scan_i16(R_max: Simd, gap: i16) -> Simd {
    let (gap_cost, _gap_cost1234, neg_inf) = get_prefix_scan_consts(gap);
    let mut curr = R_max;

    for _i in 0..(L - 1) {
        let prev = curr;
        curr = simd_sl_i16!(curr, neg_inf, 1);
        curr = i16x8_add_saturate_s(curr, gap_cost);
        curr = i16x8_max_s(curr, prev);
    }

    curr
}

#[target_feature(enable = "simd128")]
#[inline]
unsafe fn get_prefix_scan_consts(gap: i16) -> (Simd, Simd, Simd) {
    let gap_cost = _mm256_set1_epi16(gap);
    let gap_cost1234 = _mm256_set_epi16(
        gap * 4, gap * 3, gap * 2, gap * 1,
        gap * 4, gap * 3, gap * 2, gap * 1,
        gap * 4, gap * 3, gap * 2, gap * 1,
        gap * 4, gap * 3, gap * 2, gap * 1
    );
    let neg_inf = _mm256_set1_epi16(i16::MIN);
    (gap_cost, gap_cost1234, neg_inf)
}

#[target_feature(enable = "simd128")]
#[inline]
#[allow(non_snake_case)]
pub unsafe fn simd_prefix_scan_i16(R_max: Simd, gap: i16) -> Simd {
    let (gap_cost, _gap_cost1234, neg_inf) = get_prefix_scan_consts(gap);
    // Optimized prefix add and max for every four elements
    let mut shift1 = simd_sl_i16!(R_max, neg_inf, 1);
    shift1 = i16x8_add_saturate_s(shift1, gap_cost);
    shift1 = i16x8_max_s(shift1, R_max);
    let mut shift2 = simd_sl_i16!(shift1, neg_inf, 2);
    shift2 = i16x8_add_saturate_s(shift2, i16x8_shl(gap_cost, 1));
    let temp = i16x8_max_s(shift1, shift2);

    // Almost there: correct the last group using the last element of the previous group
    let mut correct = v16x8_shuffle::<0, 0, 0, 0, 3, 3, 3, 3>(temp, temp);
    correct = i16x8_add_saturate_s(correct, gap_cost1234);

    i16x8_max_s(temp, correct)
}

#[target_feature(enable = "simd128")]
#[inline]
pub unsafe fn halfsimd_lookup2_i16(lut1: HalfSimd, lut2: HalfSimd, v: HalfSimd) -> Simd {
    let mask = i8x16_splat(0b1111);
    let v_mask = v128_and(v, mask);
    let a = v8x16_swizzle(lut1, v_mask);
    let b = v8x16_swizzle(lut2, v_mask);
    let lut_mask = i8x16_gt_s(v, mask);
    let c = v128_bitselect(b, a, lut_mask);
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
pub unsafe fn halfsimd_loadu(ptr: *const HalfSimd) -> HalfSimd { v128_load(ptr) }

#[target_feature(enable = "simd128")]
#[inline]
pub unsafe fn halfsimd_store(ptr: *mut HalfSimd, a: HalfSimd) { v128_store(ptr, a) }

#[target_feature(enable = "simd128")]
#[inline]
pub unsafe fn halfsimd_sub_i8(a: HalfSimd, b: HalfSimd) -> HalfSimd { i8x16_sub(a, b) }

#[target_feature(enable = "simd128")]
#[inline]
pub unsafe fn halfsimd_set1_i8(v: i8) -> HalfSimd { i8x16_splat(v) }

// only the low 8 bytes are out of each v128 for halfsimd
#[target_feature(enable = "simd128")]
#[inline]
pub unsafe fn halfsimd_get_idx(i: usize) -> usize { i + i / L * L }

#[allow(unused_macros)]
macro_rules! halfsimd_sr_i8 {
    ($a:expr, $b:expr, $num:literal) => {
        {
            debug_assert!($num <= L);
            use std::arch::wasm32::*;
            // special indexing to skip over the high 8 bytes that are unused
            const fn get_idx(i: usize) -> usize { if i >= L { i + L } else { i } }
            v8x16_shuffle::<
                { get_idx(0 + $num) }, { get_idx(1 + $num) }, { get_idx(2 + $num) }, { get_idx(3 + $num) },
                { get_idx(4 + $num) }, { get_idx(5 + $num) }, { get_idx(6 + $num) }, { get_idx(7 + $num) },
                8, 9, 10, 11,
                12, 13, 14, 15
            >($b, $a)
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
    struct A([i8; L * HALFSIMD_MUL]);

    let mut a = A([0i8; L * HALFSIMD_MUL]);
    halfsimd_store(a.0.as_mut_ptr() as *mut HalfSimd, v);

    for i in (0..a.0.len()).rev() {
        print!("{:3} ", a.0[i]);
    }
    println!();
}

#[target_feature(enable = "simd128")]
#[allow(dead_code)]
pub unsafe fn simd_assert_vec_eq(a: Simd, b: [i16; L]) {
    #[repr(align(16))]
    struct A([i16; L]);

    let mut arr = A([0i16; L]);
    simd_store(arr.0.as_mut_ptr() as *mut Simd, a);
    assert_eq!(arr.0, b);
}

#[target_feature(enable = "simd128")]
#[allow(dead_code)]
pub unsafe fn halfsimd_assert_vec_eq(a: HalfSimd, b: [i8; L]) {
    #[repr(align(16))]
    struct A([i8; L * HALFSIMD_MUL]);

    let mut arr = A([0i8; L * HALFSIMD_MUL]);
    halfsimd_store(arr.0.as_mut_ptr() as *mut HalfSimd, a);
    assert_eq!(&arr.0[..L], b);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_endianness() {
        unsafe { test_endianness_core() };
    }

    #[target_feature(enable = "simd128")]
    unsafe fn test_endianness_core() {
        #[repr(align(16))]
        struct A([i16; L]);

        let vec = A([1, 2, 3, 4, 5, 6, 7, 8]);
        let vec = simd_load(vec.0.as_ptr() as *const Simd);
        let res = simd_sl_i16!(vec, vec, 1);
        simd_assert_vec_eq(res, [8, 1, 2, 3, 4, 5, 6, 7]);

        let vec = A([1, 2, 3, 4, 5, 6, 7, 8]);
        let vec = simd_load(vec.0.as_ptr() as *const Simd);
        let res = simd_sr_i16!(vec, vec, 1);
        simd_assert_vec_eq(res, [2, 3, 4, 5, 6, 7, 8, 1]);

        #[repr(align(16))]
        struct B([i8; L * HALFSIMD_MUL]);

        let vec = B([1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0]);
        let vec = halfsimd_load(vec.0.as_ptr() as *const HalfSimd);
        let res = halfsimd_sr_i8!(vec, vec, 1);
        halfsimd_assert_vec_eq(res, [2, 3, 4, 5, 6, 7, 8, 1]);

        let vec = simd_set4_i16(4, 3, 2, 1);
        simd_assert_vec_eq(vec, [1, 2, 3, 4, 1, 2, 3, 4]);

        simd_assert_vec_eq(simd_adds_i16(simd_set1_i16(i16::MIN), simd_set1_i16(i16::MIN)), [i16::MIN; 8]);
        simd_assert_vec_eq(simd_adds_i16(simd_set1_i16(i16::MAX), simd_set1_i16(i16::MIN)), [-1; 8]);
        simd_assert_vec_eq(simd_subs_i16(simd_set1_i16(i16::MAX), simd_set1_i16(i16::MIN)), [i16::MAX; 8]);
    }

    #[test]
    fn test_prefix_scan() {
        unsafe { test_prefix_scan_core() };
    }

    #[target_feature(enable = "simd128")]
    unsafe fn test_prefix_scan_core() {
        #[repr(align(16))]
        struct A([i16; L]);

        let vec = A([8, 9, 10, 15, 12, 13, 14, 11]);
        let res = simd_prefix_scan_i16(simd_load(vec.0.as_ptr() as *const Simd), 0);
        simd_assert_vec_eq(res, [8, 9, 10, 15, 15, 15, 15, 15]);

        let vec = A([8, 9, 10, 15, 12, 13, 14, 11]);
        let res = simd_prefix_scan_i16(simd_load(vec.0.as_ptr() as *const Simd), -1);
        simd_assert_vec_eq(res, [8, 9, 10, 15, 14, 13, 14, 13]);
    }
}
