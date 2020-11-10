#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::{alloc, mem};

const L: usize = 32usize;
const A: usize = 8usize;

// BLOSUM62 max = 11, min = -4; gap open = -11, gap extend = -1

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn scan_delta_score_avx2<M: MatrixAvx2>(database: &[u8], query: &QueryAvx2, matrix: &M) -> i32 {
    let num_vec = query.len() / L;
    let query_ptr = query.ptr();

    // R[i][j] = max(R[i - 1][j] + gap_extend, D[i - 1][j] + gap_open)
    // C[i][j] = max(C[i][j - 1] + gap_extend, D[i][j - 1] + gap_open)
    // D[i][j] = max(D[i - 1][j - 1] + matrix[query[i]][database[j]], R[i][j], C[i][j])
    //
    // indexing (we want to calculate D11):
    //      x0   x1
    //    +--------
    // 0x | 00   01
    // 1x | 10   11

    // TODO: only one array for curr?
    let delta_Dx0_ptr = alloc::alloc(alloc::Layout::from_size_align_unchecked(query.len(), L)) as *mut __m256i;
    let delta_Dx1_ptr = alloc::alloc(alloc::Layout::from_size_align_unchecked(query.len(), L)) as *mut __m256i;
    let delta_Cx0_ptr = alloc::alloc(alloc::Layout::from_size_align_unchecked(query.len(), L)) as *mut __m256i;
    let delta_Cx1_ptr = alloc::alloc(alloc::Layout::from_size_align_unchecked(query.len(), L)) as *mut __m256i;

    let num_abs = ((query.len() / L) / A) * L;
    //let abs_D_prev_ptr = alloc::alloc(alloc::Layout::from_size_align_unchecked(num_abs * 2, L)) as *mut __m256i;
    //let abs_D_curr_ptr = alloc::alloc(alloc::Layout::from_size_align_unchecked(num_abs * 2, L)) as *mut __m256i;

    let gap_open = _mm256_set1_epi8(matrix.gap_open());
    let gap_extend = _mm256_set1_epi8(matrix.gap_extend());

    for j in 0..database.len() {
        let scores = matrix.get(database[j]);

        let mut delta_D00 = _mm256_sl_epi8(delta_Dx0.offset(num_vec - 1));
        let mut delta_D10 = delta_Dx0.offset(num_vec);
        let mut delta_D01 = _mm256_set1_epi8(-128);
        let mut delta_R01 = _mm256_set1_epi8(-128);

        for i in 0..num_vec {
            let mut delta_D11 = _mm256_adds_epi8(delta_D00,
                                                 _mm256_lookup_epi8(scores, _mm256_load_si256(query_ptr.offset(i))));

            let delta_R11 = _mm256_max_epi8(_mm256_adds_epi8(delta_R01, gap_extend), _mm256_adds_epi8(delta_D01, gap_open));

            let delta_C10 = _mm256_load_si256(delta_Cx0_ptr.offset(i));
            let delta_C11 = _mm256_max_epi8(_mm256_adds_epi8(delta_C10, gap_extend), _mm256_adds_epi8(delta_D10, gap_open));

            delta_D11 = _mm256_max_epi8(delta_D11, _mm256_max_epi8(delta_R11, delta_C11));

            _mm256_store_si256(delta_Dx1_ptr.offset(i), delta_D11);
            _mm256_store_si256(delta_Cx1_ptr.offset(i), delta_C11);

            delta_D00 = delta_D10;
            delta_D10 = _mm256_load_si256(delta_Dx0.offset(i));
            delta_D01 = delta_D11;
            delta_R01 = delta_R11;
        }

        mem::swap(delta_Dx0_ptr, delta_Dx1_ptr);
        mem::swap(delta_Cx0_ptr, delta_Cx1_ptr);
    }
}

pub fn scan_delta_traceback_avx2() {

}
