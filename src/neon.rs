#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

pub type Simd = int16x8_t;
pub type HalfSimd = int16x8_t;
pub type TraceType = i32;
/// Number of 16-bit lanes in a SIMD vector.
pub const L: usize = 8;
pub const L_BYTES: usize = L * 2;
pub const HALFSIMD_MUL: usize = 2;
pub const ZERO: i16 = 1 << 14;
pub const MIN: i16 = 0;

// Non-temporal store to avoid cluttering cache with traces
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn store_trace(ptr: *mut TraceType, trace: TraceType) { *ptr = trace; }

#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn simd_adds_i16(a: Simd, b: Simd) -> Simd { vqaddq_s16(a, b) }

#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn simd_subs_i16(a: Simd, b: Simd) -> Simd { vqsubq_s16(a, b) }

#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn simd_max_i16(a: Simd, b: Simd) -> Simd { vmaxq_s16(a, b) }

#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn simd_cmpeq_i16(a: Simd, b: Simd) -> Simd { vreinterpretq_s16_u16(vceqq_s16(a, b)) }

#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn simd_cmpgt_i16(a: Simd, b: Simd) -> Simd { vreinterpretq_s16_u16(vcgtq_s16(a, b)) }

#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn simd_blend_i8(a: Simd, b: Simd, mask: Simd) -> Simd {
    let mask = vreinterpretq_u8_s8(vshrq_n_s8(vreinterpretq_s8_s16(mask), 7));
    vreinterpretq_s16_s8(vbslq_s8(mask, vreinterpretq_s8_s16(b), vreinterpretq_s8_s16(a)))
}

#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn simd_load(ptr: *const Simd) -> Simd { vreinterpretq_s16_s32(vld1q_s32(ptr as *const i32)) }

#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn simd_loadu(ptr: *const Simd) -> Simd { vreinterpretq_s16_s8(vld1q_s8(ptr as *const i8)) }

#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn simd_store(ptr: *mut Simd, a: Simd) { vst1q_s16(ptr as *mut i16, a) }

#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn simd_set1_i16(v: i16) -> Simd { vdupq_n_s16(v) }

#[macro_export]
#[doc(hidden)]
macro_rules! simd_extract_i16 {
    ($a:expr, $num:expr) => {
        {
            debug_assert!($num < L);
            #[cfg(target_arch = "aarch64")]
            use std::arch::aarch64::*;
            (vgetq_lane_s16($a, $num as i32) as u16 & (0x0000ffffu16)) as i16
        }
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! simd_insert_i16 {
    ($a:expr, $v:expr, $num:expr) => {
        {
            debug_assert!($num < L);
            #[cfg(target_arch = "aarch64")]
            use std::arch::aarch64::*;
            vsetq_lane_s16($v, $a, $num as i32)
        }
    };
}

#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn simd_movemask_i8(a: Simd) -> u32 {
    // see https://github.com/simd-everywhere/simde/blob/e1bc968696e6533d6b0bf8dddb0614737c983479/simde/x86/sse2.h#L3763
    const MD: [u8; 16] = [
        1 << 0, 1 << 1, 1 << 2, 1 << 3,
        1 << 4, 1 << 5, 1 << 6, 1 << 7,
        1 << 0, 1 << 1, 1 << 2, 1 << 3,
        1 << 4, 1 << 5, 1 << 6, 1 << 7,
    ];
    let extended = vreinterpretq_u8_s8(vshrq_n_s8(vreinterpretq_s8_s16(a), 7));
    let masked = vandq_u8(vld1q_u8(&MD[0] as *const u8), extended);
    let tmp = vzip_u8(vget_low_u8(masked), vget_high_u8(masked));
    let x = vreinterpretq_u16_u8(vcombine_u8(tmp.0, tmp.1));
    vaddvq_u16(x) as u32
 }

#[macro_export]
#[doc(hidden)]
macro_rules! simd_sl_i16 {
    ($a:expr, $b:expr, $num:expr) => {
        {
            debug_assert!(2 * $num <= L);
            #[cfg(target_arch = "aarch64")]
            use std::arch::aarch64::*;
            if $num > 31 {
                vdupq_n_s16(0)
            } else if $num > 15 {
                vreinterpretq_s16_s8(vextq_s8(vreinterpretq_s8_s16($a), vdupq_n_s8(0), ((((L * 2) - (2 * $num)) as i32) & 15)))
            } else {
                vreinterpretq_s16_s8(vextq_s8(vreinterpretq_s8_s16($b), vreinterpretq_s8_s16($a), ((((L * 2) - (2 * $num)) as i32) & 15)))
            }
        }
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! simd_sr_i16 {
    ($a:expr, $b:expr, $num:expr) => {
        {
            debug_assert!(2 * $num <= L);
            #[cfg(target_arch = "aarch64")]
            use std::arch::aarch64::*;
            if $num > 31 {
                vdupq_n_s16(0)
            } else if $num > 15 {
                vreinterpretq_s16_s8(vextq_s8(vreinterpretq_s8_s16($a), vdupq_n_s8(0), (((2 * $num) as i32) & 15)))
            } else {
                vreinterpretq_s16_s8(vextq_s8(vreinterpretq_s8_s16($b), vreinterpretq_s8_s16($a), (((2 * $num) as i32) & 15)))
            }
        }
    };
}

macro_rules! simd_sllz_i16 {
    ($a:expr, $num:expr) => {
        {
            // debug_assert!(2 * $num <= L);
            simd_sl_i16!($a, simd_set1_i16(0), $num)
        }
    };
}

#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn simd_broadcasthi_i16(v: Simd) -> Simd {
    const MASK: [u8; 16] = [14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15];
    vreinterpretq_s16_s8(vqtbl1q_s8(vreinterpretq_s8_s16(v), vld1q_u8(&MASK[0] as *const u8)))
}

#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn simd_slow_extract_i16(v: Simd, i: usize) -> i16 {
    debug_assert!(i < L);

    #[repr(align(32))]
    struct A([i16; L]);

    let mut a = A([0i16; L]);
    simd_store(a.0.as_mut_ptr() as *mut Simd, v);
    *a.0.as_ptr().add(i)
}

#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn simd_hmax_i16(v: Simd) -> i16 { vmaxvq_s16(v) }

#[macro_export]
#[doc(hidden)]
macro_rules! simd_prefix_hadd_i16 {
    ($a:expr, $num:expr) => {
        {
            debug_assert!(2 * $num <= L);
            let mut v = simd_subs_i16($a, simd_set1_i16(ZERO));
            if $num > 4 {
                v = simd_adds_i16(v, simd_sr_i16!(v, v, 4));
            }
            if $num > 2 {
                v = simd_adds_i16(v, simd_sr_i16!(v, v, 2));
            }
            if $num > 1 {
                v = simd_adds_i16(v, simd_sr_i16!(v, v, 1));
            }
            simd_extract_i16!(v, 0)
        }
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! simd_prefix_hmax_i16 {
    ($a:expr, $num:expr) => {
        {
            debug_assert!(2 * $num <= L);
            let mut v = $a;
            if $num > 4 {
                v = simd_max_i16(v, simd_sr_i16!(v, v, 4));
            }
            if $num > 2 {
                v = simd_max_i16(v, simd_sr_i16!(v, v, 2));
            }
            if $num > 1 {
                v = simd_max_i16(v, simd_sr_i16!(v, v, 1));
            }
            simd_extract_i16!(v, 0)
        }
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! simd_suffix_hmax_i16 {
    ($a:expr, $num:expr) => {
        {
            debug_assert!(2 * $num <= L);
            let mut v = $a;
            if $num > 4 {
                v = simd_max_i16(v, simd_sl_i16!(v, v, 4));
            }
            if $num > 2 {
                v = simd_max_i16(v, simd_sl_i16!(v, v, 2));
            }
            if $num > 1 {
                v = simd_max_i16(v, simd_sl_i16!(v, v, 1));
            }
            simd_extract_i16!(v, 7)
        }
    };
}

#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn simd_hargmax_i16(v: Simd, max: i16) -> usize {
    let v2 = simd_cmpeq_i16(v, simd_set1_i16(max));
    (simd_movemask_i8(v2).trailing_zeros() as usize) / 2
}

#[target_feature(enable = "neon")]
#[inline]
#[allow(non_snake_case)]
#[allow(dead_code)]
pub unsafe fn simd_naive_prefix_scan_i16(R_max: Simd, gap_cost: Simd, _gap_cost_lane: PrefixScanConsts) -> Simd {
    let mut curr = R_max;

    for _i in 0..(L - 1) {
        let prev = curr;
        curr = simd_sl_i16!(curr, vdupq_n_s16(0), 1);
        curr = simd_adds_i16(curr, gap_cost);
        curr = simd_max_i16(curr, prev);
    }

    curr
}

pub type PrefixScanConsts = Simd;

#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn get_prefix_scan_consts(gap: Simd) -> (Simd, PrefixScanConsts) {
    let mut shift1 = simd_sllz_i16!(gap, 1);
    shift1 = simd_adds_i16(shift1, gap);
    let mut shift2 = simd_sllz_i16!(shift1, 2);
    shift2 = simd_adds_i16(shift2, shift1);
    let mut shift4 = simd_sllz_i16!(shift2, 4);
    shift4 = simd_adds_i16(shift4, shift2);

    (shift4, shift4)
}

#[target_feature(enable = "neon")]
#[inline]
#[allow(non_snake_case)]
pub unsafe fn simd_prefix_scan_i16(R_max: Simd, gap_cost: Simd, _gap_cost_lane: PrefixScanConsts) -> Simd {
    // Optimized prefix add and max for every eight elements
    // Note: be very careful to avoid lane-crossing which has a large penalty.
    // Also, make sure to use as little registers as possible to avoid
    // memory loads (latencies really matter since this is critical path).
    // Keep the CPU busy with instructions!
    let mut shift1 = simd_sllz_i16!(R_max, 1);
    shift1 = simd_adds_i16(shift1, gap_cost);
    shift1 = simd_max_i16(R_max, shift1);
    let mut shift2 = simd_sllz_i16!(shift1, 2);
    shift2 = simd_adds_i16(shift2, vshlq_s16(gap_cost, vdupq_n_s16(1)));
    shift2 = simd_max_i16(shift1, shift2);
    let mut shift4 = simd_sllz_i16!(shift2, 4);
    shift4 = simd_adds_i16(shift4, vshlq_s16(gap_cost, vdupq_n_s16(2)));
    shift4 = simd_max_i16(shift2, shift4);

    shift4
}

#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn halfsimd_lookup2_i16(lut1: HalfSimd, lut2: HalfSimd, v: HalfSimd) -> Simd {
    let a = vqtbl1q_s8(vreinterpretq_s8_s16(lut1), vandq_u8(vreinterpretq_u8_s16(v), vdupq_n_u8(0x8F)));
    let b = vqtbl1q_s8(vreinterpretq_s8_s16(lut2), vandq_u8(vreinterpretq_u8_s16(v), vdupq_n_u8(0x8F)));
    let mask = vshlq_s16(v, vdupq_n_s16(3));
    let s8x16 = vreinterpretq_s8_s16(simd_blend_i8(vreinterpretq_s16_s8(a), vreinterpretq_s16_s8(b), mask));
    vmovl_s8(vget_low_s8(s8x16))
}

#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn halfsimd_lookup1_i16(lut: HalfSimd, v: HalfSimd) -> Simd {
    let s8x16 = vqtbl1q_s8(vreinterpretq_s8_s16(lut), vandq_u8(vreinterpretq_u8_s16(v), vdupq_n_u8(0x8F)));
    vmovl_s8(vget_low_s8(s8x16))
}

#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn halfsimd_lookup_bytes_i16(match_scores: HalfSimd, mismatch_scores: HalfSimd, a: HalfSimd, b: HalfSimd) -> Simd {
    let mask = vceqq_s8(vreinterpretq_s8_s16(a), vreinterpretq_s8_s16(b));
    let s8x16 = vreinterpretq_s8_s16(simd_blend_i8(mismatch_scores, match_scores, vreinterpretq_s16_u8(mask)));
    vmovl_s8(vget_low_s8(s8x16))
}

#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn halfsimd_load(ptr: *const HalfSimd) -> HalfSimd { vreinterpretq_s16_s32(vld1q_s32(ptr as *const i32)) }

#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn halfsimd_loadu(ptr: *const HalfSimd) -> HalfSimd { vreinterpretq_s16_s8(vld1q_s8(ptr as *const i8)) }

#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn halfsimd_store(ptr: *mut HalfSimd, a: HalfSimd) { vst1q_s16(ptr as *mut i16, a) }

#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn halfsimd_sub_i8(a: HalfSimd, b: HalfSimd) -> HalfSimd { vreinterpretq_s16_s8(vsubq_s8(vreinterpretq_s8_s16(a), vreinterpretq_s8_s16(b))) }

#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn halfsimd_set1_i8(v: i8) -> HalfSimd { vreinterpretq_s16_s8(vdupq_n_s8(v)) }

#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn halfsimd_get_idx(i: usize) -> usize { i }

#[macro_export]
#[doc(hidden)]
macro_rules! halfsimd_sr_i8 {
    ($a:expr, $b:expr, $num:expr) => {
        {
            debug_assert!($num <= L);
            #[cfg(target_arch = "aarch64")]
            use std::arch::aarch64::*;
            if $num > 31 {
                vdupq_n_s8(0)
            } else if $num > 15 {
                vextq_s8($a, 0 $count & 15)
            } else {
                vextq_s8($b, $a, $count & 15)
            }
        }
    };
}

#[target_feature(enable = "neon")]
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

#[target_feature(enable = "neon")]
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

#[target_feature(enable = "neon")]
#[allow(dead_code)]
pub unsafe fn simd_assert_vec_eq(a: Simd, b: [i16; L]) {
    #[repr(align(32))]
    struct A([i16; L]);

    let mut arr = A([0i16; L]);
    simd_store(arr.0.as_mut_ptr() as *mut Simd, a);
    assert_eq!(arr.0, b);
}

#[target_feature(enable = "neon")]
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
    fn test_smoke() {
        #[target_feature(enable = "neon")]
        unsafe fn inner() {
            #[repr(align(16))]
            struct A([i16; L]);

            let test = A([1, 2, 3, 4, 5, 6, 7, 8]);
            let test_rev = A([8, 7, 6, 5, 4, 3, 2, 1]);
            let vec0 = simd_load(test.0.as_ptr() as *const Simd);
            let vec0_rev = simd_load(test_rev.0.as_ptr() as *const Simd);
            let mut vec1 = simd_sl_i16!(vec0, vec0, 1);
            simd_assert_vec_eq(vec1, [8, 1, 2, 3, 4, 5, 6, 7]);

            vec1 = simd_sr_i16!(vec0, vec0, 1);
            simd_assert_vec_eq(vec1, [2, 3, 4, 5, 6, 7, 8, 1]);

            vec1 = simd_adds_i16(vec0, vec0);
            simd_dbg_i16(vec1);
            simd_assert_vec_eq(vec1, [2, 4, 6, 8, 10, 12, 14, 16]);

            vec1 = simd_subs_i16(vec0, vec0);
            simd_dbg_i16(vec1);
            simd_assert_vec_eq(vec1, [0, 0, 0, 0, 0, 0, 0, 0]);

            vec1 = simd_max_i16(vec0, vec0_rev);
            simd_dbg_i16(vec1);
            simd_assert_vec_eq(vec1, [8, 7, 6, 5, 5, 6, 7, 8]);

            vec1 = simd_cmpeq_i16(vec0, vec0_rev);
            simd_dbg_i16(vec1);
            simd_assert_vec_eq(vec1, [0, 0, 0, 0, 0, 0, 0, 0]);

            vec1 = simd_cmpeq_i16(vec0, vec0);
            simd_dbg_i16(vec1);
            simd_assert_vec_eq(vec1, [-1, -1, -1, -1, -1, -1, -1, -1]);

            vec1 = simd_cmpgt_i16(vec0, vec0_rev);
            simd_dbg_i16(vec1);
            simd_assert_vec_eq(vec1, [0, 0, 0, 0, -1, -1, -1, -1]);

            // let test_mask = A([0, 1, 0, 1, 0, 1, 0, 1]);
            // let vec0_mask = simd_load(test_mask.0.as_ptr() as *const Simd);
            // vec1 = simd_blend_i8(vec0, vec0_rev, vec0_mask);
            // simd_dbg_i16(vec1);
            // simd_assert_vec_eq(vec1, [1, 3, 3, 5, 5, 7, 7, 9]);

            let mut val = simd_extract_i16!(vec0, 0);
            assert_eq!(val, 1);

            val = simd_slow_extract_i16(vec0, 0);
            assert_eq!(val, 1);

            vec1 = simd_insert_i16!(vec0, 0, 2);
            simd_dbg_i16(vec1);
            simd_assert_vec_eq(vec1, [1, 2, 0, 4, 5, 6, 7, 8]);

            let val1 = simd_movemask_i8(vec0);
            assert_eq!(val1, 0);

            vec1 = simd_sllz_i16!(vec0, 1);
            simd_dbg_i16(vec1);
            simd_assert_vec_eq(vec1, [0, 1, 2, 3, 4, 5, 6, 7]);

            vec1 = simd_broadcasthi_i16(vec0);
            simd_dbg_i16(vec1);
            simd_assert_vec_eq(vec1, [8, 8, 8, 8, 8, 8, 8, 8]);

            val = simd_hmax_i16(vec0);
            assert_eq!(val, 8);

            val = simd_prefix_hadd_i16!(vec0, 4);
            assert_eq!(val, -32768);

            val = simd_prefix_hmax_i16!(vec0, 4);
            assert_eq!(val, 4);

            val = simd_suffix_hmax_i16!(vec0, 4);
            assert_eq!(val, 8);

            let val2 = simd_hargmax_i16(vec0, 4);
            assert_eq!(val2, 3);

            vec1 = halfsimd_lookup2_i16(vec0, vec0, vec0);
            simd_dbg_i16(vec1);
            simd_assert_vec_eq(vec1, [0, 1, 2, 1, 0, 1, 3, 1]);

            vec1 = halfsimd_lookup1_i16(vec0, vec0);
            simd_dbg_i16(vec1);
            simd_assert_vec_eq(vec1, [0, 1, 2, 1, 0, 1, 3, 1]);

            vec1 = halfsimd_lookup_bytes_i16(vec0, vec0, vec0, vec0);
            simd_dbg_i16(vec1);
            simd_assert_vec_eq(vec1, [1, 0, 2, 0, 3, 0, 4, 0]);

            vec1 = halfsimd_sub_i8(vec0, vec0);
            simd_dbg_i16(vec1);
            simd_assert_vec_eq(vec1, [0, 0, 0, 0, 0, 0, 0, 0]);

            // vec1 = halfsimd_sr_i8!(vec0, vec0, 1);
            // simd_dbg_i16(vec1);
            // simd_assert_vec_eq(vec1, [512, 768, 1024, 256, 5, 6, 7, 8]);
        }
        unsafe { inner(); }
    }

    #[test]
    fn test_prefix_scan() {
        #[target_feature(enable = "neon")]
        unsafe fn inner() {
            #[repr(align(16))]
            struct A([i16; L]);
            let vec = A([8, 9, 10, 15, 12, 13, 14, 11]);
            let gap = simd_set1_i16(0);
            let (_, consts) = get_prefix_scan_consts(gap);
            let res = simd_prefix_scan_i16(simd_load(vec.0.as_ptr() as *const Simd), gap, consts);
            simd_assert_vec_eq(res, [8, 9, 10, 15, 15, 15, 15, 15]);

            let vec = A([8, 9, 10, 15, 12, 13, 14, 11]);
            let gap = simd_set1_i16(-1);
            let (_, consts) = get_prefix_scan_consts(gap);
            let res = simd_prefix_scan_i16(simd_load(vec.0.as_ptr() as *const Simd), gap, consts);
            simd_assert_vec_eq(res, [8, 9, 10, 15, 14, 13, 14, 13]);
        }
        unsafe { inner(); }
    }
}
