use std::arch::wasm32::*;

pub type Simd = v128;
// no v64 type, so HalfSimd is just v128 with upper half ignored
pub type HalfSimd = v128;
pub type TraceType = i16;
pub const L: usize = 8;
pub const L_BYTES: usize = L * 2;
pub const HALFSIMD_MUL: usize = 2;
pub const ZERO: i16 = 1 << 14;
pub const MIN: i16 = 0;

// Note: SIMD vectors treated as little-endian

// No non-temporal store in WASM
#[inline]
pub unsafe fn store_trace(ptr: *mut TraceType, trace: TraceType) { *ptr = trace; }

#[inline]
pub unsafe fn simd_adds_i16(a: Simd, b: Simd) -> Simd { i16x8_add_sat(a, b) }

#[inline]
pub unsafe fn simd_subs_i16(a: Simd, b: Simd) -> Simd { i16x8_sub_sat(a, b) }

#[inline]
pub unsafe fn simd_max_i16(a: Simd, b: Simd) -> Simd { i16x8_max(a, b) }

#[inline]
pub unsafe fn simd_cmpeq_i16(a: Simd, b: Simd) -> Simd { i16x8_eq(a, b) }

#[inline]
pub unsafe fn simd_cmpgt_i16(a: Simd, b: Simd) -> Simd { i16x8_gt(a, b) }

#[inline]
pub unsafe fn simd_blend_i8(a: Simd, b: Simd, mask: Simd) -> Simd { v128_bitselect(b, a, mask) }

#[inline]
pub unsafe fn simd_load(ptr: *const Simd) -> Simd { v128_load(ptr) }

#[inline]
pub unsafe fn simd_store(ptr: *mut Simd, a: Simd) { v128_store(ptr, a) }

#[inline]
pub unsafe fn simd_set1_i16(v: i16) -> Simd { i16x8_splat(v) }

#[macro_export]
macro_rules! simd_extract_i16 {
    ($a:expr, $num:expr) => {
        {
            debug_assert!($num < L);
            use std::arch::wasm32::*;
            i16x8_extract_lane::<{ $num }>($a)
        }
    };
}

#[macro_export]
macro_rules! simd_insert_i16 {
    ($a:expr, $v:expr, $num:expr) => {
        {
            debug_assert!($num < L);
            use std::arch::wasm32::*;
            i16x8_replace_lane::<{ $num }>($a, $v)
        }
    };
}

#[inline]
pub unsafe fn simd_movemask_i8(a: Simd) -> u16 {
    i8x16_bitmask(a) as u16
    /*const MUL: i64 = {
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
    (res1 << 8) | res2*/
}

#[macro_export]
macro_rules! simd_sl_i16 {
    ($a:expr, $b:expr, $num:expr) => {
        {
            debug_assert!($num <= L);
            use std::arch::wasm32::*;
            i16x8_shuffle::<{ 8 - $num }, { 9 - $num }, { 10 - $num }, { 11 - $num }, { 12 - $num }, { 13 - $num }, { 14 - $num }, { 15 - $num }>($b, $a)
        }
    };
}

#[macro_export]
macro_rules! simd_sr_i16 {
    ($a:expr, $b:expr, $num:expr) => {
        {
            debug_assert!($num <= L);
            use std::arch::wasm32::*;
            i16x8_shuffle::<{ 0 + $num }, { 1 + $num }, { 2 + $num }, { 3 + $num }, { 4 + $num }, { 5 + $num }, { 6 + $num }, { 7 + $num }>($b, $a)
        }
    };
}

macro_rules! simd_sllz_i16 {
    ($a:expr, $num:expr) => {
        {
            simd_sl_i16!($a, simd_set1_i16(0), $num)
        }
    };
}

#[inline]
pub unsafe fn simd_broadcasthi_i16(v: Simd) -> Simd {
    i16x8_shuffle::<7, 7, 7, 7, 7, 7, 7, 7>(v, v)
}

#[inline]
pub unsafe fn simd_slow_extract_i16(v: Simd, i: usize) -> i16 {
    debug_assert!(i < L);

    #[repr(align(16))]
    struct A([i16; L]);

    let mut a = A([0i16; L]);
    simd_store(a.0.as_mut_ptr() as *mut Simd, v);
    *a.0.as_ptr().add(i)
}

#[inline]
pub unsafe fn simd_hmax_i16(v: Simd) -> i16 {
    let mut v2 = i16x8_max(v, simd_sr_i16!(v, v, 1));
    v2 = i16x8_max(v2, simd_sr_i16!(v2, v2, 2));
    v2 = i16x8_max(v2, simd_sr_i16!(v2, v2, 4));
    simd_extract_i16!(v2, 0)
}

#[macro_export]
macro_rules! simd_prefix_hmax_i16 {
    ($a:expr, $num:expr) => {
        {
            debug_assert!($num <= L);
            use std::arch::wasm32::*;
            let mut v = $a;
            if $num > 4 {
                v = i16x8_max(v, simd_sr_i16!(v, v, 4));
            }
            if $num > 2 {
                v = i16x8_max(v, simd_sr_i16!(v, v, 2));
            }
            if $num > 1 {
                v = i16x8_max(v, simd_sr_i16!(v, v, 1));
            }
            simd_extract_i16!(v, 0)
        }
    };
}

#[inline]
pub unsafe fn simd_hargmax_i16(v: Simd, max: i16) -> usize {
    let v2 = i16x8_eq(v, i16x8_splat(max));
    (simd_movemask_i8(v2).trailing_zeros() as usize) / 2
}

#[inline]
#[allow(non_snake_case)]
#[allow(dead_code)]
pub unsafe fn simd_naive_prefix_scan_i16(R_max: Simd, gap_cost: PrefixScanConsts) -> Simd {
    let mut curr = R_max;

    for _i in 0..(L - 1) {
        let prev = curr;
        curr = simd_sllz_i16!(curr, 1);
        curr = i16x8_add_sat(curr, gap_cost);
        curr = i16x8_max(curr, prev);
    }

    curr
}

#[inline]
pub unsafe fn get_gap_extend_all(gap: i16) -> Simd {
    i16x8(
        gap * 8, gap * 7, gap * 6, gap * 5,
        gap * 4, gap * 3, gap * 2, gap * 1
    )
}

pub type PrefixScanConsts = Simd;

#[inline]
pub unsafe fn get_prefix_scan_consts(gap: i16) -> PrefixScanConsts {
    let gap_cost = i16x8_splat(gap);
    gap_cost
}

#[inline]
#[allow(non_snake_case)]
pub unsafe fn simd_prefix_scan_i16(R_max: Simd, gap_cost: PrefixScanConsts) -> Simd {
    let mut shift1 = simd_sllz_i16!(R_max, 1);
    shift1 = i16x8_add_sat(shift1, gap_cost);
    shift1 = i16x8_max(shift1, R_max);
    let mut shift2 = simd_sllz_i16!(shift1, 2);
    shift2 = i16x8_add_sat(shift2, i16x8_shl(gap_cost, 1));
    shift2 = i16x8_max(shift1, shift2);
    let mut shift4 = simd_sllz_i16!(shift2, 4);
    shift4 = i16x8_add_sat(shift4, i16x8_shl(gap_cost, 2));
    shift4 = i16x8_max(shift2, shift4);

    shift4
}

#[inline]
pub unsafe fn halfsimd_lookup2_i16(lut1: HalfSimd, lut2: HalfSimd, v: HalfSimd) -> Simd {
    // must use a mask to avoid zeroing lanes that are too large
    let mask = i8x16_splat(0b1111);
    let v_mask = v128_and(v, mask);
    let a = i8x16_swizzle(lut1, v_mask);
    let b = i8x16_swizzle(lut2, v_mask);
    let lut_mask = i8x16_gt(v, mask);
    let c = v128_bitselect(b, a, lut_mask);
    i16x8_extend_low_i8x16(c)
}

#[inline]
pub unsafe fn halfsimd_lookup1_i16(lut: HalfSimd, v: HalfSimd) -> Simd {
    i16x8_extend_low_i8x16(i8x16_swizzle(lut, v128_and(v, i8x16_splat(0b1111))))
}

#[inline]
pub unsafe fn halfsimd_lookup_bytes_i16(match_scores: HalfSimd, mismatch_scores: HalfSimd, a: HalfSimd, b: HalfSimd) -> Simd {
    let mask = i8x16_eq(a, b);
    let c = v128_bitselect(match_scores, mismatch_scores, mask);
    i16x8_extend_low_i8x16(c)
}

#[inline]
pub unsafe fn halfsimd_load(ptr: *const HalfSimd) -> HalfSimd { v128_load(ptr) }

#[inline]
pub unsafe fn halfsimd_loadu(ptr: *const HalfSimd) -> HalfSimd { v128_load(ptr) }

#[inline]
pub unsafe fn halfsimd_store(ptr: *mut HalfSimd, a: HalfSimd) { v128_store(ptr, a) }

#[inline]
pub unsafe fn halfsimd_sub_i8(a: HalfSimd, b: HalfSimd) -> HalfSimd { i8x16_sub(a, b) }

#[inline]
pub unsafe fn halfsimd_set1_i8(v: i8) -> HalfSimd { i8x16_splat(v) }

// only the low 8 bytes are out of each v128 for halfsimd
#[inline]
pub unsafe fn halfsimd_get_idx(i: usize) -> usize { i + i / L * L }

#[macro_export]
macro_rules! halfsimd_sr_i8 {
    ($a:expr, $b:expr, $num:expr) => {
        {
            debug_assert!($num <= L);
            use std::arch::wasm32::*;
            // special indexing to skip over the high 8 bytes that are unused
            const fn get_idx(i: usize) -> usize { if i >= L { i + L } else { i } }
            i8x16_shuffle::<
                { get_idx(0 + $num) }, { get_idx(1 + $num) }, { get_idx(2 + $num) }, { get_idx(3 + $num) },
                { get_idx(4 + $num) }, { get_idx(5 + $num) }, { get_idx(6 + $num) }, { get_idx(7 + $num) },
                8, 9, 10, 11,
                12, 13, 14, 15
            >($b, $a)
        }
    };
}

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

#[allow(dead_code)]
pub unsafe fn simd_assert_vec_eq(a: Simd, b: [i16; L]) {
    #[repr(align(16))]
    struct A([i16; L]);

    let mut arr = A([0i16; L]);
    simd_store(arr.0.as_mut_ptr() as *mut Simd, a);
    assert_eq!(arr.0, b);
}

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
        unsafe {
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

            simd_assert_vec_eq(simd_adds_i16(simd_set1_i16(i16::MIN), simd_set1_i16(i16::MIN)), [i16::MIN; 8]);
            simd_assert_vec_eq(simd_adds_i16(simd_set1_i16(i16::MAX), simd_set1_i16(i16::MIN)), [-1; 8]);
            simd_assert_vec_eq(simd_subs_i16(simd_set1_i16(i16::MAX), simd_set1_i16(i16::MIN)), [i16::MAX; 8]);
        }
    }

    #[test]
    fn test_prefix_scan() {
        unsafe {
            #[repr(align(16))]
            struct A([i16; L]);

            let vec = A([8, 9, 10, 15, 12, 13, 14, 11]);
            let consts = get_prefix_scan_consts(0);
            let res = simd_prefix_scan_i16(simd_load(vec.0.as_ptr() as *const Simd), consts);
            simd_assert_vec_eq(res, [8, 9, 10, 15, 15, 15, 15, 15]);

            let vec = A([8, 9, 10, 15, 12, 13, 14, 11]);
            let consts = get_prefix_scan_consts(-1);
            let res = simd_prefix_scan_i16(simd_load(vec.0.as_ptr() as *const Simd), consts);
            simd_assert_vec_eq(res, [8, 9, 10, 15, 14, 13, 14, 13]);
        }
    }
}
