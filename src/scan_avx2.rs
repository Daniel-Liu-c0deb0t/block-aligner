#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::{alloc, mem};

use matrix::Matrix;

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

#[inline]
fn convert_char(c: u8) -> u8 {
    debug_assert!(c >= b'A' && c <= NULL);
    c - b'A'
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

/// Ensure that the reference and the query are uppercase alphabetical letters.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn scan_score_avx2(reference: &[u8], query: &[u8], matrix: &Matrix, K_half: usize) -> i32 {
    let K = K_half * 2 + 1;
    let ceil_len = ((K + L - 1) / L) * L; // round up to multiple of 16
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

    {
        let mut abs_prev = 0;

        for i in -(K_half as i32)..=(K_half as i32) {
            let idx = (i as usize) + K_half;
            let interval_idx = idx / I;
            let stride = (cmp::min(I, K - interval_idx * I) + L - 1) / L;
            let buf_idx = interval_idx * I + idx % stride;

            if i >= 0 && i < query.len() as i32 {
                ptr::write(query_buf_ptr.offset(buf_idx), convert_char(if i > 0 {
                    *query.get_unchecked(i - 1) } else { NULL }));

                let val = if i > 0 { matrix.gap_open() as i32 } else { 0 } + i * (matrix.gap_extend() as i32);

                if idx % I == 0 {
                    ptr::write(abs_Ax0_ptr.offset(interval_idx), val);
                    abs_prev = val;
                }

                ptr::write(delta_Dx0_ptr.offset(buf_idx), (val - abs_prev) as i16);
            } else {
                if idx % I == 0 {
                    ptr::write(abs_Ax0_ptr.offset(interval_idx), 0);
                }

                ptr::write(query_buf_ptr.offset(buf_idx), convert_char(NULL));
                ptr::write(delta_Dx0_ptr.offset(buf_idx), i16::MIN);
            }

            ptr::write(delta_Cx0_ptr.offset(buf_idx), i16::MIN);
        }
    }

    let query_buf_ptr = query_buf_ptr as *mut __m128i;
    let delta_Dx0_ptr = delta_Dx0_ptr as *mut __m256i;
    let delta_Cx0_ptr = delta_Cx0_ptr as *mut __m256i;

    let query_idx = K_half;
    let mut ring_buf_idx = 0;

    let gap_open = _mm256_set1_epi16(matrix.gap_open() as i16);
    let gap_extend = _mm256_set1_epi16(matrix.gap_extend() as i16);
    let neg_inf = _mm256_set1_epi16(i16::MIN);

    // TODO: x drop
    // TODO: wasm
    // TODO: adaptive banding
    // TODO: can we not save array of abs?
    // TODO: traceback

    for j in 0..reference.len() {
        let matrix_ptr = matrix.as_ptr(convert_char(*reference.get_unchecked(j)));
        let scores1 = _mm_load_si256(matrix_ptr as *const __m128i);
        let scores2 = _mm_load_si256((matrix_ptr as *const __m128i) + 1);
        let mut band_idx = 0;

        let mut abs_R_interval = i32::MIN;
        let mut abs_D_interval = i32::MIN;

        while band_idx < K {
            let stride = (cmp::min(I, K - band_idx) + L - 1) / L;

            let stride_gap_scalar = stride * gap_extend;
            let stride_gap = _mm256_set1_epi16(stride_gap_scalar);
            let mut delta_D00 = neg_inf;
            let mut abs_interval = *abs_Ax0_ptr.offset(band_idx / I);

            // Update ring buffers to slide current band down
            {
                let next_band_idx = band_idx + I;

                let delta_Dx0_idx = delta_Dx0_ptr + idx;
                // Save first vector of the previous interval before it is replaced
                delta_D00 = _mm256_load_si256(delta_Dx0_idx);
                abs_interval += _mm256_extract_epi16(delta_D00, 0) as i32;

                let query_insert;
                let delta_Dx0_insert;
                let delta_Cx0_insert;

                if next_band_idx >= K {
                    // This must be the last interval
                    let c = if query_idx < query.len() { *query.get_unchecked(query_idx) } else { NULL };
                    query_insert = _mm_set1_epi8(convert_char(c));
                    delta_Dx0_insert = neg_inf;
                    delta_Cx0_insert = neg_inf;
                } else {
                    // Not the last interval; need to shift in a value from the next interval
                    let next_stride = (cmp::min(I, K - next_band_idx) + L - 1) / L;
                    let next_idx = next_band_idx / L + ring_buf_idx % next_stride;
                    let next_abs_interval = *abs_Ax0_ptr.offset(next_band_idx / I);
                    let abs_offset = _mm256_set1_epi16((next_abs_interval - abs_interval) as i16);

                    query_insert = _mm_load_si128(query_buf_ptr + next_idx);
                    delta_Dx0_insert = _mm256_adds_epi16(_mm256_load_si256(delta_Dx0_ptr + next_idx), abs_offset);
                    delta_Cx0_insert = _mm256_adds_epi16(_mm256_load_si256(delta_Cx0_ptr + next_idx), abs_offset);
                }

                let idx = band_idx / L + ring_buf_idx % stride;
                let query_buf_idx = query_buf_ptr + idx;
                let delta_Cx0_idx = delta_Cx0_ptr + idx;

                // Now shift in new values for each interval
                _mm_store_si128(query_buf_idx, _mm_alignr_si128(query_insert, _mm_load_si128(query_buf_idx), 1));
                _mm256_store_si256(delta_Dx0_idx, _mm256_alignr_epi16(delta_Dx0_insert, delta_D00));
                _mm256_store_si256(delta_Cx0_idx, _mm256_alignr_epi16(delta_Cx0_insert, _mm256_load_si256(delta_Cx0_idx)));

                ring_buf_idx += 1;
                query_idx += 1;
            }

            // Vector for prefix scan calculations
            let mut delta_R_max = neg_inf;
            let abs_offset = _mm256_set1_epi16((*abs_Ax0_ptr.offset(band_idx / I) - abs_interval) as i16);

            // Begin initial pass
            {
                let mut delta_D01 = neg_inf;
                let mut extend_to_end = stride_gap;

                for i in 0..stride {
                    let idx = band_idx / L + (ring_buf_idx + i) % stride;

                    let scores = _mm256_lookupepi8_epi16(scores1, scores2,
                                                         _mm_load_si128(query_buf_ptr.offset(idx)));
                    let mut delta_D11 = _mm256_adds_epi16(delta_D00, scores);

                    let delta_D10 = _mm256_adds_epi16(_mm256_load_si256(delta_Dx0_ptr.offset(idx)), abs_offset);
                    let delta_C10 = _mm256_adds_epi16(_mm256_load_si256(delta_Cx0_ptr.offset(idx)), abs_offset);
                    let delta_C11 = _mm256_max_epi16(_mm256_adds_epi16(delta_C10, gap_extend), _mm256_adds_epi16(delta_D10, gap_open));

                    delta_D11 = _mm256_max_epi16(delta_D11, delta_C11);

                    extend_to_end = _mm256_subs_epi16(extend_to_end, gap_extend);
                    delta_R_max = _mm256_max_epi16(delta_R_max, _mm256_adds_epi16(delta_D11, extend_to_end));

                    // Slide band right by directly overwriting the previous band
                    _mm256_store_si256(delta_Dx0_ptr.offset(idx), delta_D11);
                    _mm256_store_si256(delta_Cx0_ptr.offset(idx), delta_C11);

                    delta_D00 = delta_D10;
                    delta_D01 = delta_D11;
                }
            }
            // End initial pass

            // Begin prefix scan
            {
                let stride_gap2 = _mm256_set1_epi16(stride_gap_scalar * 2);
                let stride_gap4 = _mm256_set1_epi16(stride_gap_scalar * 4);

                let delta_R_max_last = _mm256_extract_epi16(delta_R_max, L - 1) as i32;
                delta_R_max = _mm256_sl_epi16(delta_R_max, neg_inf);
                delta_R_max = _mm256_insert_epi16(delta_R_max, (abs_R_interval - abs_interval) as i16, 0);

                // D C B A  D C B A
                // D   B    D   B
                let reduce1 = _mm256_max_epi16(delta_R_max, _mm256_adds_epi16(stride_gap, _mm256_slli_si256(delta_R_max, 2)));
                // D        D
                let reduce2 = _mm256_max_epi16(reduce1, _mm256_adds_epi16(stride_gap2, _mm256_slli_si256(reduce1, 4)));

                for _ in 0..(L / 4 - 1) {
                    let prev = reduce2;
                    reduce2 = _mm256_sl_epi64(reduce2, neg_inf);
                    reduce2 = _mm256_adds_epi16(stride_gap4);
                    reduce2 = _mm256_max_epi16(reduce2, prev);
                }
                // reduce2
                // D        D

                //     B        B
                let unreduce1_mid = _mm256_max_epi16(_mm256_adds_epi16(_mm256_sl_epi32(reduce2, neg_inf), stride_gap2), reduce1);
                // D   B    D   B
                let unreduce1 = _mm256_blend_epi16(reduce2, unreduce1_mid, 0b0010_0010_0010_0010);
                //   C   A    C   A
                let unreduce2 = _mm256_max_epi16(_mm256_adds_epi16(_mm256_sl_epi16(unreduce1, neg_inf), stride_gap), delta_R_max);
                // D C B A  D C B A
                delta_R_max = _mm256_blend_epi16(unreduce1, unreduce2, 0b0101_0101_0101_0101);

                abs_R_interval = abs_interval + cmp::max(delta_R_max_last, _mm256_extract_epi16(_mm256_adds_epi16(delta_R_max, stride_gap), L - 1) as i32);
            }
            // End prefix scan

            // Begin final pass
            {
                let mut delta_R01 = _mm256_subs_epi16(_mm256_adds_epi16(delta_R_max, gap_extend), gap_open);
                let mut delta_D01 = _mm256_insert_epi16(neg_inf, (abs_D_interval - abs_interval) as i16, 0);

                for i in 0..stride {
                    let idx = band_idx / L + (ring_buf_idx + i) % stride;

                    let mut delta_R11 = _mm256_max_epi16(_mm256_adds_epi16(delta_R01, gap_extend), _mm256_adds_epi16(delta_D01, gap_open));
                    let mut delta_D11 = _mm256_load_si256(delta_Dx0_ptr.offset(idx));
                    delta_D11 = _mm256_max_epi16(delta_D11, delta_R11);

                    _mm256_store_si256(delta_Dx0_ptr.offset(idx), delta_D11);

                    delta_D01 = delta_D11;
                    delta_R01 = delta_R11;
                }

                abs_D_interval = abs_interval + (_mm256_extract_epi16(delta_D01, L - 1) as i32);
            }
            // End final pass

            *abs_Ax0_ptr.offset(band_idx / I) = abs_interval;
            band_idx += I;
        }
    }

    alloc::dealloc(query_buf_ptr as *mut u8, query_buf_layout);
    alloc::dealloc(delta_Dx0_ptr as *mut u8, delta_Dx0_layout);
    alloc::dealloc(delta_Cx0_ptr as *mut u8, delta_Cx0_layout);
    alloc::dealloc(abs_Ax0_ptr as *mut u8, abs_Ax0_layout);
}

pub fn scan_traceback_avx2() {

}
