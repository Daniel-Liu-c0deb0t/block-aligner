#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::alloc;

const L: usize = 32usize;
const A: usize = 8usize;

// BLOSUM62 max = 11, min = -4; gap open = -11, gap extend = -1

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn scan_delta_score_avx2<M: MatrixAvx2>(database: &[u8], query: &QueryAvx2, matrix: &M) -> i32 {
    let num_vec = query.len() / L;
    let query_ptr = query.ptr();

    // TODO: only one array for curr?
    let curr_col_delta_ptr = alloc::alloc(alloc::Layout::from_size_align_unchecked(query.len(), L)) as *mut __m256i;
    let prev_col_delta_ptr = alloc::alloc(alloc::Layout::from_size_align_unchecked(query.len(), L)) as *mut __m256i;

    let num_abs = ((query.len() / L) / A) * L;
    let curr_col_abs_ptr = alloc::alloc(alloc::Layout::from_size_align_unchecked(num_abs * 2, L)) as *mut __m256i;
    let prev_col_abs_ptr = alloc::alloc(alloc::Layout::from_size_align_unchecked(num_abs * 2, L)) as *mut __m256i;

    let gap_open = _mm256_set1_epi8(matrix.gap_open());
    let gap_extend = _mm256_set1_epi8(matrix.gap_extend());

    for i in 0..database.len() {
        let curr_scores = matrix.get(database[i]);

        let mut prev_diag = _mm256_sl_epi8(prev_col_delta_ptr.offset(num_vec - 1));
        let mut prev_row = _mm256_set1_epi8(-128);

        for j in 0..num_vec {
            let prev_col = _mm256_load_si256(prev_col_delta_ptr);

            let curr_diag = _mm256_adds_epi8(prev_diag,
                                             _mm256_lookup_epi8(curr_scores, _mm256_load_si256(query_ptr.offset(j))));

            let curr_row = _mm256_max_epi8(_mm256_adds_epi8(prev_row, gap_extend), _mm256_adds_epi8(prev_row, gap_open));
            let curr_col = _mm256_max_epi8(_mm256_adds_epi8(prev_col, gap_extend), _mm256_adds_epi8(prev_col, gap_open));

            prev_diag = curr_col;
            prev_row = curr_row;
        }
    }
}

pub fn scan_delta_traceback_avx2() {

}
