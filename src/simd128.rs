use std::arch::wasm32::*;

use std::cmp;

pub type Simd = v128;
pub type HalfSimd = v128;
pub const L: usize = 8;
pub const L_BYTES: usize = L * 2;

// TODO: example folder without dependencies
// TODO: simd indexing order?

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
pub unsafe fn simd_set4_i16(d: i16, c: i16, b: i16, a: i16) -> Simd { i16x8_const(d, c, b, a, d, c, b, a) }

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
pub unsafe fn simd_prefix_scan_i16(delta_R_max: Simd, stride_gap: Simd, stride_gap1234: Simd, neg_inf: Simd) -> Simd {
    // Optimized prefix add and max for every four elements
    let mut shift1 = simd_sl_i16!(delta_R_max, neg_inf, 1);
    shift1 = i16x8_add_saturate_s(shift1, stride_gap);
    shift1 = i16x8_max_s(shift1, delta_R_max);
    let mut shift2 = simd_sl_i16!(shift1, neg_inf, 2);
    shift2 = i16x8_add_saturate_s(shift2, i16x8_shl(stride_gap, 1));
    let temp = i16x8_max_s(shift1, shift2);

    // Almost there: correct the last group using the last element of the previous group
    let mut correct = v16x8_shuffle::<3, 3, 3, 3, 0, 0, 0, 0>(temp, temp);
    correct = i16x8_add_saturate_s(correct, stride_gap1234);

    i16x8_max_s(temp, correct)
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
pub unsafe fn halfsimd_set1_i8(v: i8) -> HalfSimd { i8x16_splat(v) }

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
        let res = simd_prefix_scan_i16(simd_load(vec.0.as_ptr() as *const Simd), simd_set1_i16(0), simd_set4_i16(0, 0, 0, 0), simd_set1_i16(i16::MIN));
        simd_assert_vec_eq(res, [8, 9, 10, 15, 15, 15, 15, 15]);

        let vec = A([8, 9, 10, 15, 12, 13, 14, 11]);
        let res = simd_prefix_scan_i16(simd_load(vec.0.as_ptr() as *const Simd), simd_set1_i16(-1), simd_set4_i16(-4, -3, -2, -1), simd_set1_i16(i16::MIN));
        simd_assert_vec_eq(res, [8, 9, 10, 15, 14, 13, 14, 13]);
    }
}
