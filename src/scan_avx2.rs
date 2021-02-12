#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::{alloc, cmp, ptr, i16};
use std::marker::PhantomData;

use crate::scores::*;

const L: usize = 16usize;
const L_BYTES: usize = L * 2;
const NULL: u8 = b'A' + 26u8; // this null byte value works for both amino acids and nucleotides

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn _mm256_lookup32_epi16(lut1: __m128i, lut2: __m128i, v: __m128i) -> __m256i {
    let a = _mm_shuffle_epi8(lut1, v);
    let b = _mm_shuffle_epi8(lut2, v);
    let mask = _mm_cmpgt_epi8(_mm_set1_epi8(0b00010000), v);
    let c = _mm_blendv_epi8(b, a, mask);
    _mm256_cvtepi8_epi16(c)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn _mm256_lookup16_epi16(lut: __m128i, v: __m128i) -> __m256i {
    _mm256_cvtepi8_epi16(_mm_shuffle_epi8(lut, v))
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
fn convert_char(c: u8, nuc: bool) -> u8 {
    debug_assert!(c >= b'A' && c <= NULL);
    if nuc { c } else { c - b'A' }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hmax(mut v: __m256i) -> i16 {
    v = _mm256_max_epi16(v, _mm256_srli_si256(v, 2));
    v = _mm256_max_epi16(v, _mm256_srli_si256(v, 4));
    v = _mm256_max_epi16(v, _mm256_srli_si256(v, 8));
    cmp::max(_mm256_extract_epi16(v, 0) as i16, _mm256_extract_epi16(v, (L as i32) / 2) as i16)
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

// BLOSUM62 matrix max = 11, min = -4; gap open = -11 (includes extension), gap extend = -1
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
//
// TODO: abstract into class for step by step
// TODO: x drop and early exit
// TODO: adaptive banding
// TODO: wasm
// TODO: i8

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(non_snake_case)]
pub struct ScanAligner<'a, P: ScoreParams, M: 'a + Matrix, const K_HALF: usize, const TRACE: bool> {
    query_buf_layout: alloc::Layout,
    query_buf_ptr: *mut __m128i,
    delta_Dx0_layout: alloc::Layout,
    delta_Dx0_ptr: *mut __m256i,
    delta_Cx0_layout: alloc::Layout,
    delta_Cx0_ptr: *mut __m256i,
    abs_Ax0_layout: alloc::Layout,
    abs_Ax0_ptr: *mut i32,

    trace: Vec<u32>,

    query_idx: usize,
    shift_idx: isize,
    ring_buf_idx: usize,

    query: &'a [u8],
    matrix: &'a M,

    _phantom: PhantomData<P>
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl<'a, P: ScoreParams, M: 'a + Matrix, const K_HALF: usize, const TRACE: bool> ScanAligner<'a, P, M, { K_HALF }, { TRACE }> {
    const K: usize = K_HALF * 2 + 1;
    const CEIL_K: usize = ((Self::K + L - 1) / L) * L; // round up to multiple of L
    const NUM_INTERVALS: usize = (Self::CEIL_K + P::I - 1) / P::I;

    // Use precomputed strides so compiler can avoid division/modulo instructions
    const STRIDE_I: usize = P::I / L;
    const STRIDE_LAST: usize = (Self::CEIL_K - ((Self::CEIL_K - 1) / P::I) * P::I) / L;

    const EVEN_BITS: u32 = 0x55555555u32;

    #[target_feature(enable = "avx2")]
    #[allow(non_snake_case)]
    pub unsafe fn new(query: &'a [u8], matrix: &'a M) -> Self {
        assert!(P::GAP_OPEN <= P::GAP_EXTEND);
        assert!(P::I % L == 0);

        // These chunks of memory are contiguous ring buffers that represent every interval in the current band
        let query_buf_layout = alloc::Layout::from_size_align_unchecked(Self::CEIL_K, L_BYTES);
        let query_buf_ptr = alloc::alloc(query_buf_layout) as *mut u8;

        let delta_Dx0_layout = alloc::Layout::from_size_align_unchecked(Self::CEIL_K * 2, L_BYTES);
        let delta_Dx0_ptr = alloc::alloc(delta_Dx0_layout) as *mut i16;

        let delta_Cx0_layout = alloc::Layout::from_size_align_unchecked(Self::CEIL_K * 2, L_BYTES);
        let delta_Cx0_ptr = alloc::alloc(delta_Cx0_layout) as *mut i16;

        // 32-bit absolute values
        let abs_Ax0_layout = alloc::Layout::array::<i32>(Self::NUM_INTERVALS).unwrap();
        let abs_Ax0_ptr = alloc::alloc(abs_Ax0_layout) as *mut i32;

        // Initialize DP columns
        // Not extremely optimized, since it only runs once
        {
            let mut abs_prev = 0;

            for idx in 0..Self::CEIL_K {
                let i = (idx as isize) - (K_HALF as isize);
                let interval_idx = idx / P::I;
                let stride = cmp::min(P::I, Self::CEIL_K - interval_idx * P::I) / L;
                let buf_idx = interval_idx * P::I + (((idx % P::I) % stride) * L + (idx % P::I) / stride);
                debug_assert!(buf_idx < Self::CEIL_K);

                if i >= 0 && i <= query.len() as isize {
                    ptr::write(query_buf_ptr.add(buf_idx), convert_char(if i > 0 {
                        *query.get_unchecked(i as usize - 1) } else { NULL }, M::NUC));

                    let val = if i > 0 { (P::GAP_OPEN as i32) + ((i as i32) - 1) * (P::GAP_EXTEND as i32) } else { 0 };

                    if idx % P::I == 0 {
                        ptr::write(abs_Ax0_ptr.add(interval_idx), val);
                        abs_prev = val;
                    }

                    ptr::write(delta_Dx0_ptr.add(buf_idx), (val - abs_prev) as i16);
                } else {
                    if idx % P::I == 0 {
                        ptr::write(abs_Ax0_ptr.add(interval_idx), 0);
                    }

                    ptr::write(query_buf_ptr.add(buf_idx), convert_char(NULL, M::NUC));
                    ptr::write(delta_Dx0_ptr.add(buf_idx), i16::MIN);
                }

                ptr::write(delta_Cx0_ptr.add(buf_idx), i16::MIN);
            }
        }

        Self {
            query_buf_layout,
            query_buf_ptr: query_buf_ptr as *mut __m128i,
            delta_Dx0_layout,
            delta_Dx0_ptr: delta_Dx0_ptr as *mut __m256i,
            delta_Cx0_layout,
            delta_Cx0_ptr: delta_Cx0_ptr as *mut __m256i,
            abs_Ax0_layout,
            abs_Ax0_ptr,

            trace: vec![],

            query_idx: Self::CEIL_K - K_HALF - 1,
            shift_idx: -(K_HALF as isize),
            ring_buf_idx: 0,

            query,
            matrix,

            _phantom: PhantomData
        }
    }

    /// Banded alignment.
    ///
    /// Limitations:
    /// 1. Requires AVX2 support.
    /// 2. The reference and the query can only contain uppercase alphabetical characters.
    /// 3. The actual size of the band is K_HALF * 2 + 1 rounded up to the next multiple of the
    ///    vector length of 16.
    #[target_feature(enable = "avx2")]
    #[allow(non_snake_case)]
    pub unsafe fn align(&mut self, reference: &[u8]) {
        // optional 32-bit traceback
        // 0b00 = up and left, 0b10 or 0b11 = up, 0b01 = left
        if TRACE {
            self.trace.resize(self.trace.len() + (reference.len() + 1) * Self::CEIL_K / L, Self::EVEN_BITS << 1);
        }

        let gap_open = _mm256_set1_epi16(P::GAP_OPEN as i16);
        let gap_extend = _mm256_set1_epi16(P::GAP_EXTEND as i16);
        let neg_inf = _mm256_set1_epi16(i16::MIN);

        for j in 0..reference.len() {
            // Load scores for the current reference character
            let matrix_ptr = self.matrix.as_ptr(convert_char(*reference.get_unchecked(j), M::NUC) as usize);
            let scores1 = _mm_load_si128(matrix_ptr as *const __m128i);
            let scores2 = if M::NUC { None } else { Some(_mm_load_si128((matrix_ptr as *const __m128i).add(1))) };

            let mut band_idx = 0usize;
            let mut abs_R_interval = i16::MIN as i32;
            let mut abs_D_interval = i16::MIN as i32;

            while band_idx < Self::CEIL_K {
                let last_interval = (band_idx + P::I) >= Self::CEIL_K;
                let stride = if last_interval { Self::STRIDE_LAST } else { Self::STRIDE_I };
                let stride_gap = _mm256_set1_epi16((stride as i16) * (P::GAP_EXTEND as i16));
                let mut delta_D00;
                let mut abs_interval = *self.abs_Ax0_ptr.add(band_idx / P::I);

                // Update ring buffers to slide current band down
                {
                    let idx = band_idx / L
                        + if last_interval { self.ring_buf_idx % Self::STRIDE_LAST } else { self.ring_buf_idx % Self::STRIDE_I };
                    let delta_Dx0_idx = self.delta_Dx0_ptr.add(idx);
                    // Save first vector of the previous interval before it is replaced
                    delta_D00 = _mm256_load_si256(delta_Dx0_idx);

                    if self.shift_idx + (band_idx as isize) >= 0 {
                        abs_interval = abs_interval.saturating_add(_mm256_extract_epi16(delta_D00, 0) as i16 as i32);
                    }

                    let query_buf_idx = self.query_buf_ptr.add(idx);
                    let delta_Cx0_idx = self.delta_Cx0_ptr.add(idx);

                    if last_interval {
                        // This must be the last interval
                        let c = if self.query_idx < self.query.len() { *self.query.get_unchecked(self.query_idx) } else { NULL };
                        let query_insert = _mm_set1_epi8(convert_char(c, M::NUC) as i8);

                        // Now shift in new values for each interval
                        _mm_store_si128(query_buf_idx, _mm_alignr_epi8(query_insert, _mm_load_si128(query_buf_idx), 1));
                        _mm256_store_si256(delta_Dx0_idx, _mm256_alignr_epi16(neg_inf, delta_D00));
                        _mm256_store_si256(delta_Cx0_idx, _mm256_alignr_epi16(neg_inf, _mm256_load_si256(delta_Cx0_idx)));
                    } else {
                        // Not the last interval; need to shift in a value from the next interval
                        let next_band_idx = band_idx + P::I;
                        let next_last_interval = next_band_idx >= Self::CEIL_K;
                        let next_idx = next_band_idx / L +
                            if next_last_interval { self.ring_buf_idx % Self::STRIDE_LAST } else { self.ring_buf_idx % Self::STRIDE_I };
                        let next_abs_interval = *self.abs_Ax0_ptr.add(next_band_idx / P::I);
                        let abs_offset = _mm256_set1_epi16(clamp(next_abs_interval - abs_interval));
                        debug_assert!(next_idx < Self::CEIL_K / L);

                        let query_insert = _mm_load_si128(self.query_buf_ptr.add(next_idx));
                        let delta_Dx0_insert = _mm256_adds_epi16(_mm256_load_si256(self.delta_Dx0_ptr.add(next_idx)), abs_offset);
                        let delta_Cx0_insert = _mm256_adds_epi16(_mm256_load_si256(self.delta_Cx0_ptr.add(next_idx)), abs_offset);

                        // Now shift in new values for each interval
                        _mm_store_si128(query_buf_idx, _mm_alignr_epi8(query_insert, _mm_load_si128(query_buf_idx), 1));
                        _mm256_store_si256(delta_Dx0_idx, _mm256_alignr_epi16(delta_Dx0_insert, delta_D00));
                        _mm256_store_si256(delta_Cx0_idx, _mm256_alignr_epi16(delta_Cx0_insert, _mm256_load_si256(delta_Cx0_idx)));
                    }
                }

                // Vector for prefix scan calculations
                let mut delta_R_max = neg_inf;
                let abs_offset = _mm256_set1_epi16(clamp(*self.abs_Ax0_ptr.add(band_idx / P::I) - abs_interval));
                delta_D00 = _mm256_adds_epi16(delta_D00, abs_offset);

                // Begin initial pass
                {
                    let mut extend_to_end = stride_gap;

                    for i in 0..stride {
                        let idx = {
                            let mut idx = self.ring_buf_idx + 1 + i;
                            idx = if last_interval { idx % Self::STRIDE_LAST } else { idx % Self::STRIDE_I };
                            band_idx / L + idx
                        };
                        debug_assert!(idx < Self::CEIL_K / L);

                        let scores = if M::NUC {
                            _mm256_lookup16_epi16(scores1, _mm_load_si128(self.query_buf_ptr.add(idx)))
                        } else {
                            _mm256_lookup32_epi16(scores1, scores2.unwrap(), _mm_load_si128(self.query_buf_ptr.add(idx)))
                        };

                        let mut delta_D11 = _mm256_adds_epi16(delta_D00, scores);

                        let delta_D10 = _mm256_adds_epi16(_mm256_load_si256(self.delta_Dx0_ptr.add(idx)), abs_offset);
                        let delta_C10 = _mm256_adds_epi16(_mm256_load_si256(self.delta_Cx0_ptr.add(idx)), abs_offset);
                        let delta_C11 = _mm256_max_epi16(_mm256_adds_epi16(delta_C10, gap_extend), _mm256_adds_epi16(delta_D10, gap_open));

                        delta_D11 = _mm256_max_epi16(delta_D11, delta_C11);

                        if TRACE {
                            let trace_idx = (Self::CEIL_K / L) * (j + 1) + band_idx / L + i;
                            debug_assert!(trace_idx < self.trace.len());
                            *self.trace.get_unchecked_mut(trace_idx) = _mm256_movemask_epi8(_mm256_cmpeq_epi16(delta_C11, delta_D11)) as u32;
                        }

                        extend_to_end = _mm256_subs_epi16(extend_to_end, gap_extend);
                        delta_R_max = _mm256_max_epi16(delta_R_max, _mm256_adds_epi16(delta_D11, extend_to_end));

                        // Slide band right by directly overwriting the previous band
                        _mm256_store_si256(self.delta_Dx0_ptr.add(idx), delta_D11);
                        _mm256_store_si256(self.delta_Cx0_ptr.add(idx), delta_C11);

                        delta_D00 = delta_D10;
                    }
                }
                // End initial pass

                // Begin prefix scan
                {
                    let prev_delta_R_max_last = _mm256_extract_epi16(delta_R_max, L as i32 - 1) as i16 as i32;
                    delta_R_max = _mm256_sl_epi16(delta_R_max, neg_inf);
                    delta_R_max = _mm256_insert_epi16(delta_R_max, clamp(abs_R_interval - abs_interval), 0);

                    delta_R_max = prefix_scan_epi16(delta_R_max, stride_gap, neg_inf);

                    let curr_delta_R_max_last = _mm256_extract_epi16(_mm256_adds_epi16(delta_R_max, stride_gap), L as i32 - 1) as i16 as i32;
                    abs_R_interval = abs_interval.saturating_add(cmp::max(prev_delta_R_max_last, curr_delta_R_max_last));
                }
                // End prefix scan

                // Begin final pass
                {
                    let mut delta_R01 = _mm256_adds_epi16(_mm256_subs_epi16(delta_R_max, gap_extend), gap_open);
                    let mut delta_D01 = _mm256_insert_epi16(neg_inf, clamp(abs_D_interval - abs_interval), 0);

                    for i in 0..stride {
                        let idx = {
                            let mut idx = self.ring_buf_idx + 1 + i;
                            idx = if last_interval { idx % Self::STRIDE_LAST } else { idx % Self::STRIDE_I };
                            band_idx / L + idx
                        };
                        debug_assert!(idx < Self::CEIL_K / L);

                        let delta_R11 = _mm256_max_epi16(_mm256_adds_epi16(delta_R01, gap_extend), _mm256_adds_epi16(delta_D01, gap_open));
                        let mut delta_D11 = _mm256_load_si256(self.delta_Dx0_ptr.add(idx));
                        delta_D11 = _mm256_max_epi16(delta_D11, delta_R11);

                        if TRACE {
                            let trace_idx = (Self::CEIL_K / L) * (j + 1) + band_idx / L + i;
                            debug_assert!(trace_idx < self.trace.len());
                            let prev_trace = *self.trace.get_unchecked(trace_idx);
                            let curr_trace = _mm256_movemask_epi8(_mm256_cmpeq_epi16(delta_R11, delta_D11)) as u32;
                            *self.trace.get_unchecked_mut(trace_idx) = (prev_trace & Self::EVEN_BITS) | ((curr_trace & Self::EVEN_BITS) << 1);
                        }

                        _mm256_store_si256(self.delta_Dx0_ptr.add(idx), delta_D11);

                        delta_D01 = delta_D11;
                        delta_R01 = delta_R11;
                    }

                    abs_D_interval = abs_interval.saturating_add(_mm256_extract_epi16(delta_D01, L as i32 - 1) as i16 as i32);
                }
                // End final pass

                debug_assert!(band_idx / P::I < Self::NUM_INTERVALS);
                *self.abs_Ax0_ptr.add(band_idx / P::I) = abs_interval;
                band_idx += P::I;
            }

            self.ring_buf_idx += 1;
            self.query_idx += 1;
            self.shift_idx += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn score(&self) -> i32 {
        // Extract the score from the last band
        let res_i = ((self.query.len() as isize) - self.shift_idx) as usize;
        let band_idx = (res_i / P::I) * P::I;
        let stride = cmp::min(P::I, Self::CEIL_K - band_idx) / L;
        let idx = band_idx / L + (res_i % P::I) % stride;
        debug_assert!(idx < Self::CEIL_K / L);

        let delta = slow_extract_epi16(_mm256_load_si256(self.delta_Dx0_ptr.add(idx)), (res_i % P::I) / stride) as i32;
        let abs = *self.abs_Ax0_ptr.add(res_i / P::I);

        delta + abs
    }

    pub fn raw_trace(&self) -> &[u32] {
        assert!(TRACE);
        &self.trace
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl<'a, P: ScoreParams, M: 'a + Matrix, const K_HALF: usize, const TRACE: bool> Drop for ScanAligner<'a, P, M, { K_HALF }, { TRACE }> {
    fn drop(&mut self) {
        unsafe {
            alloc::dealloc(self.query_buf_ptr as *mut u8, self.query_buf_layout);
            alloc::dealloc(self.delta_Dx0_ptr as *mut u8, self.delta_Dx0_layout);
            alloc::dealloc(self.delta_Cx0_ptr as *mut u8, self.delta_Cx0_layout);
            alloc::dealloc(self.abs_Ax0_ptr as *mut u8, self.abs_Ax0_layout);
        }
    }
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
        type TestParams = Params<-11, -1, 1024>;

        let r = b"AAAA";
        let q = b"AARA";
        let mut a = ScanAligner::<TestParams, _, 1, false>::new(q, &BLOSUM62);
        a.align(r);
        assert_eq!(a.score(), 11);

        let r = b"AAAA";
        let q = b"AARA";
        let mut a = ScanAligner::<TestParams, _, 3, false>::new(q, &BLOSUM62);
        a.align(r);
        assert_eq!(a.score(), 11);

        let r = b"AAAA";
        let q = b"AAAA";
        let mut a = ScanAligner::<TestParams, _, 1, false>::new(q, &BLOSUM62);
        a.align(r);
        assert_eq!(a.score(), 16);

        let r = b"AAAA";
        let q = b"AARA";
        let mut a = ScanAligner::<TestParams, _, 0, false>::new(q, &BLOSUM62);
        a.align(r);
        assert_eq!(a.score(), 11);

        let r = b"AAAA";
        let q = b"RRRR";
        let mut a = ScanAligner::<TestParams, _, 4, false>::new(q, &BLOSUM62);
        a.align(r);
        assert_eq!(a.score(), -4);

        let r = b"AAAA";
        let q = b"AAA";
        let mut a = ScanAligner::<TestParams, _, 1, false>::new(q, &BLOSUM62);
        a.align(r);
        assert_eq!(a.score(), 1);

        type TestParams2 = Params<-1, -1, 2048>;

        let r = b"AAAN";
        let q = b"ATAA";
        let mut a = ScanAligner::<TestParams2, _, 2, false>::new(q, &NW1);
        a.align(r);
        assert_eq!(a.score(), 1);

        let r = b"AAAA";
        let q = b"C";
        let mut a = ScanAligner::<TestParams2, _, 4, false>::new(q, &NW1);
        a.align(r);
        assert_eq!(a.score(), -4);
        let mut a = ScanAligner::<TestParams2, _, 4, false>::new(r, &NW1);
        a.align(q);
        assert_eq!(a.score(), -4);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn assert_vec_eq(a: __m256i, b: [i16; L]) {
        let mut arr = [0i16; L];
        _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, a);
        assert_eq!(arr, b);
    }
}
