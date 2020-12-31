#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::{alloc, mem};

const L: usize = 16usize;
const L_BYTES: usize = 32usize;
const NULL: u8 = b'A' + 31u8;
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
unsafe fn _mm256_sl_epi16(v: __m256i) -> __m256i {
    _mm256_alignr_epi8(v, _mm256_permute2x128_si256(v, v, 0x0F), 14)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn _mm256_sr_epi16(v: __m256i) -> __m256i {
    _mm256_alignr_epi8(_mm256_permute2x128_si256(v, v, 0xF1), v, 2)
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
unsafe fn _mm256_sl_epi32(v: __m256i) -> __m256i {
    _mm256_alignr_epi8(v, _mm256_permute2x128_si256(v, v, 0x0F), 12)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn _mm256_sl_epi64(v: __m256i) -> __m256i {
    _mm256_alignr_epi8(v, _mm256_permute2x128_si256(v, v, 0x0F), 8)
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn scan_score_avx2<M: Matrix>(reference: &[u8], query: &QueryAvx2, matrix: &M, K: usize) -> i32 {
    let ceil_len = ((K + L - 1) / L) * L; // round up to multiple of 16
    let ceil_len_bytes = ceil_len * 2;
    let num_intervals = (K + I - 1) / I;

    // These chunks of memory are contiguous ring buffers that represent every interval in the current band
    let query_buf_ptr = alloc::alloc(alloc::Layout::from_size_align_unchecked(ceil_len, L_BYTES)) as *mut __m128i;
    let delta_Dx0_ptr = alloc::alloc(alloc::Layout::from_size_align_unchecked(ceil_len_bytes, L_BYTES)) as *mut __m256i;
    let delta_Cx0_ptr = alloc::alloc(alloc::Layout::from_size_align_unchecked(ceil_len_bytes, L_BYTES)) as *mut __m256i;

    // 32-bit absolute values
    let abs_Ax0_ptr = alloc::alloc(alloc::Layout::from_size_align_unchecked(num_intervals * 4, 4)) as *mut i32;

    let mut ring_buf_idx = 0;

    let query_idx = 0; // TODO

    let gap_open = _mm256_set1_epi16(matrix.gap_open() as i16);
    let gap_extend = _mm256_set1_epi16(matrix.gap_extend() as i16);

    // TODO: x drop
    // TODO: wasm
    // TODO: adaptive banding
    // TODO: vector of i16::MIN
    // TODO: can we not save array of abs?

    for j in 0..reference.len() {
        let matrix_ptr = matrix.ptr(*reference.get_unchecked(j));
        let scores1 = _mm_load_si256(matrix_ptr as *const __m128i);
        let scores2 = _mm_load_si256((matrix_ptr as *const __m128i) + 1);
        let mut band_idx = 0;

        let mut abs_R_interval = i32::MIN;
        let mut abs_D_interval = i32::MIN;

        while band_idx < K {
            let stride = (cmp::min(I, K - band_idx) + L - 1) / L;

            let stride_gap_scalar = stride * gap_extend;
            let stride_gap = _mm256_set1_epi16(stride_gap_scalar);
            let mut delta_D00 = _mm256_set1_epi16(i16::MIN);
            let mut abs_interval = *abs_Ax0_ptr.offset(band_idx / I);

            // Update ring buffers to slide current band down
            if j > 0 {
                let query_insert;
                let delta_Dx0_insert;
                let delta_Cx0_insert;
                let next_band_idx = band_idx + I;

                let delta_Dx0_idx = delta_Dx0_ptr + idx;
                // Save first vector of the previous interval before it is replaced
                delta_D00 = _mm256_load_si256(delta_Dx0_idx);
                abs_interval += _mm256_extract_epi16(delta_D00, 0) as i32;

                if next_band_idx >= K {
                    // This must be the last interval
                    if query_idx < 0 || query_idx >= query.len() {
                        query_insert = _mm_set1_epi8(NULL);
                    } else {
                        query_insert = _mm_set1_epi8(*query.get_unchecked(query_idx));
                    }

                    delta_Dx0_insert = _mm256_set1_epi16(i16::MIN);
                    delta_Cx0_insert = _mm256_set1_epi16(i16::MIN);
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
            }

            // Vector for prefix scan calculations
            let mut delta_R_max = _mm256_set1_epi16(i16::MIN);
            let abs_offset = _mm256_set1_epi16((*abs_Ax0_ptr.offset(band_idx / I) - abs_interval) as i16);

            // Begin initial pass
            {
                let mut delta_D01 = _mm256_set1_epi16(i16::MIN);
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
                delta_R_max = _mm256_sl_epi16(delta_R_max);
                delta_R_max = _mm256_insert_epi16(delta_R_max, (abs_R_interval - abs_interval) as i16, 0);

                // D C B A  D C B A
                // D   B    D   B
                let reduce1 = _mm256_max_epi16(delta_R_max, _mm256_adds_epi16(stride_gap, _mm256_slli_si256(delta_R_max, 2)));
                // D        D
                let reduce2 = _mm256_max_epi16(reduce1, _mm256_adds_epi16(stride_gap2, _mm256_slli_si256(reduce1, 4)));

                for _ in 0..(L / 4 - 1) {
                    let prev = reduce2;
                    reduce2 = _mm256_sl_epi64(reduce2);
                    reduce2 = _mm256_insert_epi16(reduce2, i16::MIN, 0);
                    reduce2 = _mm256_adds_epi16(stride_gap4);
                    reduce2 = _mm256_max_epi16(reduce2, prev);
                }
                // reduce2
                // D        D

                //     B        B
                let unreduce1_mid = _mm256_max_epi16(_mm256_adds_epi16(_mm256_insert_epi32(_mm256_sl_epi32(reduce2), 0x8080u32 as i32, 0), stride_gap2), reduce1);
                // D   B    D   B
                let unreduce1 = _mm256_blend_epi16(reduce2, unreduce1_mid, 0b0010_0010_0010_0010);
                //   C   A    C   A
                let unreduce2 = _mm256_max_epi16(_mm256_adds_epi16(_mm256_insert_epi16(_mm256_sl_epi16(unreduce1), i16::MIN, 0), stride_gap), delta_R_max);
                // D C B A  D C B A
                delta_R_max = _mm256_blend_epi16(unreduce1, unreduce2, 0b0101_0101_0101_0101);

                abs_R_interval = abs_interval + cmp::max(delta_R_max_last, _mm256_extract_epi16(_mm256_adds_epi16(delta_R_max, stride_gap), L - 1) as i32);
            }
            // End prefix scan

            // Begin final pass
            {
                let mut delta_R01 = _mm256_subs_epi16(_mm256_adds_epi16(delta_R_max, gap_extend), gap_open);
                let mut delta_D01 = _mm256_insert_epi16(_mm256_set1_epi16(i16::MIN), (abs_D_interval - abs_interval) as i16, 0);

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

        // Finish sliding current band down
        if j > 0 {
            ring_buf_idx += 1;
            query_idx += 1;
        }
    }
}

pub fn scan_traceback_avx2() {

}
