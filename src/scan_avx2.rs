#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::{alloc, mem};

const L: usize = 16usize;

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
    _mm256_alignr_epi8(v, _mm256_permute2x128_si256(v, v, 0x03), 14)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn _mm256_sl_epi32(v: __m256i) -> __m256i {
    _mm256_alignr_epi8(v, _mm256_permute2x128_si256(v, v, 0x03), 12)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn _mm256_sl_epi64(v: __m256i) -> __m256i {
    _mm256_alignr_epi8(v, _mm256_permute2x128_si256(v, v, 0x03), 8)
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn scan_score_avx2<M: Matrix>(reference: &[u8], query: &QueryAvx2, matrix: &M) -> i32 {
    let num_vec = query.len() / L;
    let query_ptr = query.ptr();

    // TODO: only one array for curr?
    let delta_Dx0_ptr = alloc::alloc(alloc::Layout::from_size_align_unchecked(query.len(), L)) as *mut __m256i;
    let delta_Dx1_ptr = alloc::alloc(alloc::Layout::from_size_align_unchecked(query.len(), L)) as *mut __m256i;
    let delta_Cx0_ptr = alloc::alloc(alloc::Layout::from_size_align_unchecked(query.len(), L)) as *mut __m256i;
    let delta_Cx1_ptr = alloc::alloc(alloc::Layout::from_size_align_unchecked(query.len(), L)) as *mut __m256i;

    let gap_open = _mm256_set1_epi16(matrix.gap_open());
    let gap_extend = _mm256_set1_epi16(matrix.gap_extend());
    let num_vec_gap = _mm256_set1_epi16(query.len() / L * gap_extend);
    let num_vec_gap2 = _mm256_set1_epi16(query.len() / L * gap_extend * 2);
    let num_vec_gap4 = _mm256_set1_epi16(query.len() / L * gap_extend * 4);

    for j in 0..reference.len() {
        let matrix_ptr = matrix.ptr(reference[j]);
        let scores1 = _mm_load_si256(matrix_ptr as *const __m128i);
        let scores2 = _mm_load_si256((matrix_ptr as *const __m128i) + 1);

        let mut delta_R_max = _mm256_set1_epi16(i16::MIN);

        // Begin initial pass
        {
            let mut delta_D00 = _mm256_sl_epi16(delta_Dx0.offset(num_vec - 1)); // TODO: insert
            let mut delta_D10 = delta_Dx0.offset(num_vec);
            let mut delta_D01 = _mm256_set1_epi16(i16::MIN);
            let mut extend_to_end = num_vec_gap;

            for i in 0..num_vec {
                let scores = _mm256_lookupepi8_epi16(scores1, scores2, _mm256_load_si256(query_ptr.offset(i)));
                let mut delta_D11 = _mm256_adds_epi16(delta_D00, scores);

                let delta_C10 = _mm256_load_si256(delta_Cx0_ptr.offset(i));
                let delta_C11 = _mm256_max_epi16(_mm256_adds_epi16(delta_C10, gap_extend), _mm256_adds_epi16(delta_D10, gap_open));

                delta_D11 = _mm256_max_epi16(delta_D11, delta_C11);

                extend_to_end = _mm256_subs_epi16(extend_to_end, gap_extend);
                delta_R_max = _mm256_max_epi16(delta_R_max, _mm256_adds_epi16(delta_D11, extend_to_end));

                _mm256_store_si256(delta_Dx1_ptr.offset(i), delta_D11);
                _mm256_store_si256(delta_Cx1_ptr.offset(i), delta_C11);

                delta_D00 = delta_D10;
                delta_D10 = _mm256_load_si256(delta_Dx0.offset(i));
                delta_D01 = delta_D11;
            }
        }
        // End initial pass

        // Begin prefix scan
        {
            delta_R_max = _mm256_sl_epi16(delta_R_max);
            delta_R_max = _mm256_insert_epi16(delta_R_max, i16::MIN, 0);

            // D C B A  D C B A
            // D   B    D   B
            let reduce1 = _mm256_max_epi16(delta_R_max, _mm256_adds_epi16(num_vec_gap, _mm256_slli_si256(delta_R_max, 2)));
            // D        D
            let reduce2 = _mm256_max_epi16(reduce1, _mm256_adds_epi16(num_vec_gap2, _mm256_slli_si256(reduce1, 4)));

            for _ in 0..(L / 4 - 1) {
                let prev = reduce2;
                reduce2 = _mm256_sl_epi64(reduce2);
                reduce2 = _mm256_insert_epi16(reduce2, i16::MIN, 0);
                reduce2 = _mm256_adds_epi16(num_vec_gap4);
                reduce2 = _mm256_max_epi16(reduce2, prev);
            }
            // reduce2
            // D        D

            //     B        B
            let unreduce1_mid = _mm256_max_epi16(_mm256_adds_epi16(_mm256_insert_epi32(_mm256_sl_epi32(reduce2), 0x8080u32 as i32, 0), num_vec_gap2), reduce1);
            // D   B    D   B
            let unreduce1 = _mm256_blend_epi16(reduce2, unreduce1_mid, 0b0010_0010_0010_0010);
            //   C   A    C   A
            let unreduce2 = _mm256_max_epi16(_mm256_adds_epi16(_mm256_insert_epi16(_mm256_sl_epi16(unreduce1), i16::MIN, 0), num_vec_gap), delta_R_max);
            // D C B A  D C B A
            delta_R_max = _mm256_blend_epi16(unreduce1, unreduce2, 0b0101_0101_0101_0101);
        }
        // End prefix scan

        // Begin final pass
        {
            let mut delta_R01 = _mm256_subs_epi16(_mm256_adds_epi16(delta_R_max, gap_extend), gap_open);
            let mut delta_D01 = _mm256_set1_epi16(i16::MIN);

            for i in 0..num_vec {
                let mut delta_R11 = _mm256_max_epi16(_mm256_adds_epi16(delta_R01, gap_extend), _mm256_adds_epi16(delta_D01, gap_open));
                let mut delta_D11 = _mm256_load_si256(delta_Dx1_ptr.offset(i));
                delta_D11 = _mm256_max_epi16(delta_D11, delta_R11);

                _mm256_store_si256(delta_Dx1_ptr.offset(i), delta_D11);

                delta_D01 = delta_D11;
                delta_R01 = delta_R11;
            }
        }
        // End final pass

        mem::swap(&mut delta_Dx0_ptr, &mut delta_Dx1_ptr);
        mem::swap(&mut delta_Cx0_ptr, &mut delta_Cx1_ptr);
    }
}

pub fn scan_traceback_avx2() {

}
