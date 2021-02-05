#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::intrinsics::unlikely;

use std::{alloc, cmp, ptr, i16};

use crate::scores::*;

const L: usize = 16usize;
const L_BYTES: usize = 32usize;
const NULL: u8 = b'A' + 26u8;
const I: usize = 1024usize; // I % L == 0

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn _mm256_lookupepi8_epi16(lut1: __m128i, lut2: __m128i, v: __m128i) -> __m256i {
    let a = _mm_shuffle_epi8(lut1, v);
    let b = _mm_shuffle_epi8(lut2, v);
    let mask = _mm_cmpgt_epi8(_mm_set1_epi8(0b00010000), v);
    let c = _mm_blendv_epi8(b, a, mask);
    _mm256_cvtepi8_epi16(c)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn _mm256_sl_epi16(v: __m256i, zeros: __m256i) -> __m256i {
    _mm256_alignr_epi8(v, _mm256_permute2x128_si256(v, zeros, 0x02), 14)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn _mm256_alignr_epi16(a: __m256i, b: __m256i) -> __m256i {
    _mm256_alignr_epi8(_mm256_permute2x128_si256(a, b, 0x03), b, 2)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn _mm256_sl_epi32(v: __m256i, zeros: __m256i) -> __m256i {
    _mm256_alignr_epi8(v, _mm256_permute2x128_si256(v, zeros, 0x02), 12)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn _mm256_sl_epi64(v: __m256i, zeros: __m256i) -> __m256i {
    _mm256_alignr_epi8(v, _mm256_permute2x128_si256(v, zeros, 0x02), 8)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn _mm256_sl_epi128(v: __m256i, zeros: __m256i) -> __m256i {
    _mm256_permute2x128_si256(v, zeros, 0x02)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn _mm256_sl_epi192(v: __m256i, zeros: __m256i) -> __m256i {
    _mm256_alignr_epi8(_mm256_permute2x128_si256(v, zeros, 0x02), zeros, 8)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn slow_extract_epi16(v: __m256i, i: usize) -> i16 {
    debug_assert!(i < L);

    #[repr(align(32))]
    struct A([i16; L]);

    let mut a = A([0i16; L]);
    _mm256_store_si256(a.0.as_mut_ptr() as *mut __m256i, v);
    *a.0.get_unchecked(i)
}

#[inline]
fn convert_char(c: u8) -> u8 {
    debug_assert!(c >= b'A' && c <= NULL);
    c - b'A'
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hmax(mut v: __m256i) -> i16 {
    v = _mm256_max_epi16(v, _mm256_srli_si256(v, 2));
    v = _mm256_max_epi16(v, _mm256_srli_si256(v, 4));
    v = _mm256_max_epi16(v, _mm256_srli_si256(v, 8));
    cmp::max(_mm256_extract_epi16(v, 0), _mm256_extract_epi16(v, (L as i32) / 2))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(non_snake_case)]
unsafe fn prefix_scan_epi16(delta_R_max: __m256i, stride_gap: __m256i, neg_inf: __m256i) -> __m256i {
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
        let mut shift4 = _mm256_sl_epi64(reduce2, neg_inf);
        let mut shift8 = _mm256_sl_epi128(reduce2, neg_inf);
        let mut shift12 = _mm256_sl_epi192(reduce2, neg_inf);

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

    let unreduce1_mid = _mm256_max_epi16(_mm256_adds_epi16(_mm256_sl_epi32(reduce2, neg_inf), stride_gap2), reduce1);
    //     B        B        B        B
    let unreduce1 = _mm256_blend_epi16(reduce2, unreduce1_mid, 0b0010_0010_0010_0010);
    // D   B    D   B    D   B    D   B
    let unreduce2 = _mm256_max_epi16(_mm256_adds_epi16(_mm256_sl_epi16(unreduce1, neg_inf), stride_gap), delta_R_max);
    //   C   A    C   A    C   A    C   A
    _mm256_blend_epi16(unreduce1, unreduce2, 0b0101_0101_0101_0101)
    // D C B A  D C B A  D C B A  D C B A
}

#[inline]
fn clamp(x: i32) -> i16 {
    cmp::min(cmp::max(x, i16::MIN as i32), i16::MAX as i32) as i16
}

// BLOSUM62 matrix max = 11, min = -4; gap open = -11, gap extend = -1
//
// R[i][j] = max(R[i - 1][j] + gap_extend, D[i - 1][j] + gap_open)
// C[i][j] = max(C[i][j - 1] + gap_extend, D[i][j - 1] + gap_open)
// D[i][j] = max(D[i - 1][j - 1] + matrix[query[i]][reference[j]], R[i][j], C[i][j])
//
// indexing (we want to calculate D11):
//      x0   x1
//    +--------
// 0x | 00   01
// 1x | 10   11
//
// A band consists of multiple intervals. Each interval is made up of strided vectors.

/// Banded alignment.
///
/// Limitations:
/// 1. Requires AVX2 support.
/// 2. The reference and the query can only contain uppercase alphabetical characters.
/// 3. The actual size of the band is K_HALF * 2 + 1 rounded up to the next multiple of the
///    vector length of 16.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(non_snake_case)]
pub unsafe fn scan_align<G: GapScores, const K_HALF: usize, const TRACE: bool, const NUC: bool>(reference: &[u8], query: &[u8], matrix: &SubMatrix) -> (i32, Option<Vec<u32>>) {
    let K = K_HALF * 2 + 1;
    let ceil_len = ((K + L - 1) / L) * L; // round up to multiple of L
    let ceil_len_bytes = ceil_len * 2;
    let num_intervals = (K + I - 1) / I;

    // These chunks of memory are contiguous ring buffers that represent every interval in the current band
    let query_buf_layout = alloc::Layout::from_size_align_unchecked(ceil_len, L_BYTES);
    let query_buf_ptr = alloc::alloc(query_buf_layout) as *mut u8;

    let delta_Dx0_layout = alloc::Layout::from_size_align_unchecked(ceil_len_bytes, L_BYTES);
    let delta_Dx0_ptr = alloc::alloc(delta_Dx0_layout) as *mut i16;

    let delta_Cx0_layout = alloc::Layout::from_size_align_unchecked(ceil_len_bytes, L_BYTES);
    let delta_Cx0_ptr = alloc::alloc(delta_Cx0_layout) as *mut i16;

    // 32-bit absolute values
    let abs_Ax0_layout = alloc::Layout::from_size_align_unchecked(num_intervals * 4, 4);
    let abs_Ax0_ptr = alloc::alloc(abs_Ax0_layout) as *mut i32;

    // optional 32-bit traceback
    // 0b00 = up and left, 0b10 or 0b11 = up, 0b01 = left
    let mut trace;
    let even_bits = 0x55555555u32;

    if TRACE {
        trace = vec![even_bits << 1; (reference.len() + 1) * ceil_len / L];
    } else {
        trace = vec![];
    }

    {
        let mut abs_prev = 0;

        for idx in 0..ceil_len {
            let i = (idx as isize) - (K_HALF as isize);
            let interval_idx = idx / I;
            let stride = (cmp::min(I, K - interval_idx * I) + L - 1) / L;
            let buf_idx = interval_idx * I + ((idx % stride) * L + idx / stride);

            if i >= 0 && i <= query.len() as isize {
                ptr::write(query_buf_ptr.add(buf_idx), convert_char(if i > 0 {
                    *query.get_unchecked(i as usize - 1) } else { NULL }));

                let val = if i > 0 { (G::GAP_OPEN as i32) + ((i as i32) - 1) * (G::GAP_EXTEND as i32) } else { 0 };

                if idx % I == 0 {
                    ptr::write(abs_Ax0_ptr.add(interval_idx), val);
                    abs_prev = val;
                }

                ptr::write(delta_Dx0_ptr.add(buf_idx), (val - abs_prev) as i16);
            } else {
                if idx % I == 0 {
                    ptr::write(abs_Ax0_ptr.add(interval_idx), 0);
                }

                ptr::write(query_buf_ptr.add(buf_idx), convert_char(NULL));
                ptr::write(delta_Dx0_ptr.add(buf_idx), i16::MIN);
            }

            ptr::write(delta_Cx0_ptr.add(buf_idx), i16::MIN);
        }
    }

    let query_buf_ptr = query_buf_ptr as *mut __m128i;
    let delta_Dx0_ptr = delta_Dx0_ptr as *mut __m256i;
    let delta_Cx0_ptr = delta_Cx0_ptr as *mut __m256i;

    let mut query_idx = ceil_len - K_HALF - 1;
    let mut shift_idx = -(K_HALF as isize);
    let mut ring_buf_idx = 0usize;

    let gap_open = _mm256_set1_epi16(G::GAP_OPEN as i16);
    let gap_extend = _mm256_set1_epi16(G::GAP_EXTEND as i16);
    let neg_inf = _mm256_set1_epi16(i16::MIN);

    // TODO: x drop
    // TODO: wasm
    // TODO: adaptive banding
    // TODO: faster support for nucleotides and const I
    // TODO: split interval loop into 2, one handles interval len I, one handles special case OR
    // adaptive I
    // get rid of division/modulo!!!
    // TODO: abstract into class for step by step
    // TODO: early exit??

    for j in 0..reference.len() {
        let matrix_ptr = matrix.as_ptr(convert_char(*reference.get_unchecked(j)) as usize);
        let scores1 = _mm_load_si128(matrix_ptr as *const __m128i);
        let scores2 = _mm_load_si128((matrix_ptr as *const __m128i).add(1));
        let mut band_idx = 0usize;

        let mut abs_R_interval = *abs_Ax0_ptr;
        let mut abs_D_interval = *abs_Ax0_ptr;

        while band_idx < K {
            let stride = (cmp::min(I, K - band_idx) + L - 1) / L;

            let stride_gap = _mm256_set1_epi16((stride as i16) * (G::GAP_EXTEND as i16));
            let mut delta_D00;
            let mut abs_interval = *abs_Ax0_ptr.add(band_idx / I);

            // Update ring buffers to slide current band down
            {
                let next_band_idx = band_idx + I;

                let idx = band_idx / L + ring_buf_idx % stride;
                let delta_Dx0_idx = delta_Dx0_ptr.add(idx);
                // Save first vector of the previous interval before it is replaced
                delta_D00 = _mm256_load_si256(delta_Dx0_idx);

                if shift_idx + (band_idx as isize) >= 0 {
                    abs_interval = abs_interval.saturating_add(_mm256_extract_epi16(delta_D00, 0) as i32);
                }

                let query_buf_idx = query_buf_ptr.add(idx);
                let delta_Cx0_idx = delta_Cx0_ptr.add(idx);

                if next_band_idx >= K {
                    // This must be the last interval
                    let c = if query_idx < query.len() { *query.get_unchecked(query_idx) } else { NULL };
                    let query_insert = _mm_set1_epi8(convert_char(c) as i8);

                    // Now shift in new values for each interval
                    _mm_store_si128(query_buf_idx, _mm_alignr_epi8(query_insert, _mm_load_si128(query_buf_idx), 1));
                    _mm256_store_si256(delta_Dx0_idx, _mm256_alignr_epi16(neg_inf, delta_D00));
                    _mm256_store_si256(delta_Cx0_idx, _mm256_alignr_epi16(neg_inf, _mm256_load_si256(delta_Cx0_idx)));
                } else {
                    // Not the last interval; need to shift in a value from the next interval
                    let next_stride = (cmp::min(I, K - next_band_idx) + L - 1) / L;
                    let next_idx = next_band_idx / L + ring_buf_idx % next_stride;
                    let next_abs_interval = *abs_Ax0_ptr.add(next_band_idx / I);
                    let abs_offset = _mm256_set1_epi16(clamp(next_abs_interval - abs_interval));

                    let query_insert = _mm_load_si128(query_buf_ptr.add(next_idx));
                    let delta_Dx0_insert = _mm256_adds_epi16(_mm256_load_si256(delta_Dx0_ptr.add(next_idx)), abs_offset);
                    let delta_Cx0_insert = _mm256_adds_epi16(_mm256_load_si256(delta_Cx0_ptr.add(next_idx)), abs_offset);

                    // Now shift in new values for each interval
                    _mm_store_si128(query_buf_idx, _mm_alignr_epi8(query_insert, _mm_load_si128(query_buf_idx), 1));
                    _mm256_store_si256(delta_Dx0_idx, _mm256_alignr_epi16(delta_Dx0_insert, delta_D00));
                    _mm256_store_si256(delta_Cx0_idx, _mm256_alignr_epi16(delta_Cx0_insert, _mm256_load_si256(delta_Cx0_idx)));
                }
            }

            // Vector for prefix scan calculations
            let mut delta_R_max = neg_inf;
            let abs_offset = _mm256_set1_epi16(clamp(*abs_Ax0_ptr.add(band_idx / I) - abs_interval));
            delta_D00 = _mm256_adds_epi16(delta_D00, abs_offset);

            // Begin initial pass
            {
                let mut extend_to_end = stride_gap;

                for i in 0..stride {
                    let idx = band_idx / L + (ring_buf_idx + 1 + i) % stride;

                    let scores = _mm256_lookupepi8_epi16(scores1, scores2,
                                                         _mm_load_si128(query_buf_ptr.add(idx)));
                    let mut delta_D11 = _mm256_adds_epi16(delta_D00, scores);

                    let delta_D10 = _mm256_adds_epi16(_mm256_load_si256(delta_Dx0_ptr.add(idx)), abs_offset);
                    let delta_C10 = _mm256_adds_epi16(_mm256_load_si256(delta_Cx0_ptr.add(idx)), abs_offset);
                    let delta_C11 = _mm256_max_epi16(_mm256_adds_epi16(delta_C10, gap_extend), _mm256_adds_epi16(delta_D10, gap_open));

                    delta_D11 = _mm256_max_epi16(delta_D11, delta_C11);

                    if TRACE {
                        let trace_idx = (ceil_len / L) * (j + 1) + band_idx / L + i;
                        *trace.get_unchecked_mut(trace_idx) = _mm256_movemask_epi8(_mm256_cmpeq_epi16(delta_C11, delta_D11)) as u32;
                    }

                    extend_to_end = _mm256_subs_epi16(extend_to_end, gap_extend);
                    delta_R_max = _mm256_max_epi16(delta_R_max, _mm256_adds_epi16(delta_D11, extend_to_end));

                    // Slide band right by directly overwriting the previous band
                    _mm256_store_si256(delta_Dx0_ptr.add(idx), delta_D11);
                    _mm256_store_si256(delta_Cx0_ptr.add(idx), delta_C11);

                    delta_D00 = delta_D10;
                }
            }
            // End initial pass

            // Begin prefix scan
            {
                let delta_R_max_last = _mm256_extract_epi16(delta_R_max, L as i32 - 1) as i32;
                delta_R_max = _mm256_sl_epi16(delta_R_max, neg_inf);
                delta_R_max = _mm256_insert_epi16(delta_R_max, clamp(abs_R_interval - abs_interval), 0);

                delta_R_max = prefix_scan_epi16(delta_R_max, stride_gap, neg_inf);

                abs_R_interval = abs_interval.saturating_add(cmp::max(delta_R_max_last, _mm256_extract_epi16(_mm256_adds_epi16(delta_R_max, stride_gap), L as i32 - 1) as i32));
            }
            // End prefix scan

            // Begin final pass
            {
                let mut delta_R01 = _mm256_adds_epi16(_mm256_subs_epi16(delta_R_max, gap_extend), gap_open);
                let mut delta_D01 = _mm256_insert_epi16(neg_inf, clamp(abs_D_interval - abs_interval), 0);

                for i in 0..stride {
                    let idx = band_idx / L + (ring_buf_idx + 1 + i) % stride;

                    let delta_R11 = _mm256_max_epi16(_mm256_adds_epi16(delta_R01, gap_extend), _mm256_adds_epi16(delta_D01, gap_open));
                    let mut delta_D11 = _mm256_load_si256(delta_Dx0_ptr.add(idx));
                    delta_D11 = _mm256_max_epi16(delta_D11, delta_R11);

                    if TRACE {
                        let trace_idx = (ceil_len / L) * (j + 1) + band_idx / L + i;
                        let prev_trace = *trace.get_unchecked(trace_idx);
                        let curr_trace = _mm256_movemask_epi8(_mm256_cmpeq_epi16(delta_R11, delta_D11)) as u32;
                        *trace.get_unchecked_mut(trace_idx) = (prev_trace & even_bits) | ((curr_trace & even_bits) << 1);
                    }

                    _mm256_store_si256(delta_Dx0_ptr.add(idx), delta_D11);

                    delta_D01 = delta_D11;
                    delta_R01 = delta_R11;
                }

                abs_D_interval = abs_interval.saturating_add(_mm256_extract_epi16(delta_D01, L as i32 - 1) as i32);
            }
            // End final pass

            *abs_Ax0_ptr.add(band_idx / I) = abs_interval;
            band_idx += I;
        }

        ring_buf_idx += 1;
        query_idx += 1;
        shift_idx += 1;
    }

    // Extract the score from the last band
    let res_score = {
        let res_i = ((query.len() as isize) - shift_idx) as usize;
        let band_idx = (res_i / I) * I;
        let stride = (cmp::min(I, K - band_idx) + L - 1) / L;
        let idx = band_idx / L + (res_i % I) % stride;

        let delta = slow_extract_epi16(_mm256_load_si256(delta_Dx0_ptr.add(idx)), (res_i % I) / stride) as i32;
        let abs = *abs_Ax0_ptr.add(res_i / I);

        delta + abs
    };

    alloc::dealloc(query_buf_ptr as *mut u8, query_buf_layout);
    alloc::dealloc(delta_Dx0_ptr as *mut u8, delta_Dx0_layout);
    alloc::dealloc(delta_Cx0_ptr as *mut u8, delta_Cx0_layout);
    alloc::dealloc(abs_Ax0_ptr as *mut u8, abs_Ax0_layout);

    if TRACE { (res_score, Some(trace)) } else { (res_score, None) }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(dead_code)]
unsafe fn dbg_avx(v: __m256i) {
    let mut a = [0i16; L];
    _mm256_storeu_si256(a.as_mut_ptr() as *mut __m256i, v);

    for i in (0..a.len()).rev() {
        print!("{:6} ", a[i]);
    }
    println!();
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(dead_code)]
unsafe fn dbg_sse(v: __m128i) {
    let mut a = [0i8; L];
    _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, v);

    for i in (0..a.len()).rev() {
        print!("{:2} ", a[i]);
    }
    println!();
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    use crate::scores::*;

    use super::*;

    #[test]
    fn test_prefix_scan() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { test_prefix_scan_core() };
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn test_prefix_scan_core() {
        let vec = _mm256_set_epi16(11, 14, 13, 12, 15, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let res = prefix_scan_epi16(vec, _mm256_setzero_si256(), _mm256_set1_epi16(i16::MIN));
        assert_vec_eq(res, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 15, 15, 15, 15]);

        let vec = _mm256_set_epi16(11, 14, 13, 12, 15, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let res = prefix_scan_epi16(vec, _mm256_set1_epi16(-1), _mm256_set1_epi16(i16::MIN));
        assert_vec_eq(res, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 14, 13, 14, 13]);
    }

    #[test]
    fn test_scan_align() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { test_scan_align_core() };
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn test_scan_align_core() {
        type Scores = Gap<-11, -1>;

        let r = b"AAAA";
        let q = b"AARA";
        let res = scan_align::<Scores, 1, false, false>(r, q, &BLOSUM62);
        assert_eq!(res.0, 11);

        let r = b"AAAA";
        let q = b"AARA";
        let res = scan_align::<Scores, 3, false, false>(r, q, &BLOSUM62);
        assert_eq!(res.0, 11);

        let r = b"AAAA";
        let q = b"AAAA";
        let res = scan_align::<Scores, 1, false, false>(r, q, &BLOSUM62);
        assert_eq!(res.0, 16);

        let r = b"AAAA";
        let q = b"AARA";
        let res = scan_align::<Scores, 0, false, false>(r, q, &BLOSUM62);
        assert_eq!(res.0, 11);

        let r = b"AAAA";
        let q = b"RRRR";
        let res = scan_align::<Scores, 4, false, false>(r, q, &BLOSUM62);
        assert_eq!(res.0, -4);

        let r = b"AAAA";
        let q = b"AAA";
        let res = scan_align::<Scores, 1, false, false>(r, q, &BLOSUM62);
        assert_eq!(res.0, 1);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn assert_vec_eq(a: __m256i, b: [i16; L]) {
        let mut arr = [0i16; L];
        _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, a);
        assert_eq!(arr, b);
    }
}
