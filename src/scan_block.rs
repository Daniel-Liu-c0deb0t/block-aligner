#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"))]
use crate::avx2::*;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::simd128::*;

use crate::scores::*;
use crate::cigar::*;

use std::intrinsics::unlikely;
use std::{cmp, ptr, i16, alloc};
use std::ops::RangeInclusive;

// Notes:
//
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
// note that 'x' represents any bit
//
// Each block is made up of vertical SIMD vectors of length 8 or 16 16-bit integers.

pub struct Block<'a, M: 'a + Matrix, const TRACE: bool, const X_DROP: bool> {
    res: AlignResult,
    trace: Trace,
    query: &'a PaddedBytes,
    i: usize,
    reference: &'a PaddedBytes,
    j: usize,
    min_size: usize,
    max_size: usize,
    matrix: &'a M,
    gaps: Gaps,
    x_drop: i32
}

// increasing step size gives a bit extra speed but results in lower accuracy
const STEP: usize = 4;
const LARGE_STEP: usize = 4; // use larger step size when the block size gets large
const GROW_STEP: usize = L; // used when not growing by powers of 2
const GROW_EXP: bool = true; // grow by powers of 2
impl<'a, M: 'a + Matrix, const TRACE: bool, const X_DROP: bool> Block<'a, M, { TRACE }, { X_DROP }> {
    /// Adaptive banded alignment.
    ///
    /// The x drop option indicates whether to terminate the alignment process early when
    /// the max score in the current band drops below the max score encountered so far. If
    /// x drop is not enabled, then the band will keep shifting until the end of the reference
    /// string is reached.
    ///
    /// Limitations:
    /// 1. Requires x86 AVX2 or WASM SIMD support.
    /// 2. The reference and the query can only contain uppercase alphabetical characters.
    /// 3. The actual size of the band is K + 1 rounded up to the next multiple of the
    ///    vector length of 16 (for x86 AVX2) or 8 (for WASM SIMD).
    pub fn align(query: &'a PaddedBytes, reference: &'a PaddedBytes, matrix: &'a M, gaps: Gaps, size: RangeInclusive<usize>, x_drop: i32) -> Self {
        // performance is not as good if there is no gap open cost
        assert!(gaps.open <= gaps.extend);
        let min_size = if *size.start() < L { L } else { *size.start() };
        let max_size = if *size.end() < L { L } else { *size.end() };
        assert!(min_size % L == 0 && max_size % L == 0);

        if X_DROP {
            assert!(x_drop >= 0);
        }

        let mut a = Self {
            res: AlignResult { score: 0, query_idx: 0, reference_idx: 0 },
            trace: if TRACE { Trace::new(query.len(), reference.len(), max_size) } else { Trace::new(0, 0, 0) },
            query,
            i: 0,
            reference,
            j: 0,
            min_size,
            max_size,
            matrix,
            gaps,
            x_drop
        };

        unsafe { a.align_core(); }
        a
    }

    #[allow(non_snake_case)]
    unsafe fn align_core(&mut self) {
        let mut best_max = 0i32;
        let mut best_argmax_i = 0usize;
        let mut best_argmax_j = 0usize;

        let mut dir = Direction::Grow;
        let mut prev_size = 0;
        let mut block_size = self.min_size;
        let mut step = STEP;

        let mut off = 0i32;
        let mut prev_off;

        let mut D_col = Aligned::new(self.max_size);
        let mut C_col = Aligned::new(self.max_size);
        let mut D_row = Aligned::new(self.max_size);
        let mut R_row = Aligned::new(self.max_size);

        let mut temp_buf1 = Aligned::new(L);
        let mut temp_buf2 = Aligned::new(L);

        let mut y_drop_iter = 0;
        let mut i_ckpt = self.i;
        let mut j_ckpt = self.j;
        let mut off_ckpt = 0i32;
        let mut D_col_ckpt = Aligned::new(self.max_size);
        let mut C_col_ckpt = Aligned::new(self.max_size);
        let mut D_row_ckpt = Aligned::new(self.max_size);
        let mut R_row_ckpt = Aligned::new(self.max_size);
        let mut i_ckpt2 = self.i;
        let mut j_ckpt2 = self.j;
        let mut off_ckpt2 = 0i32;
        let mut D_col_ckpt2 = Aligned::new(self.max_size);
        let mut C_col_ckpt2 = Aligned::new(self.max_size);
        let mut D_row_ckpt2 = Aligned::new(self.max_size);
        let mut R_row_ckpt2 = Aligned::new(self.max_size);

        loop {
            #[cfg(feature = "debug")]
            {
                println!("i: {}", self.i);
                println!("j: {}", self.j);
                println!("{:?}", dir);
                println!("block size: {}", block_size);
            }

            prev_off = off;
            let mut grow_D_max = simd_set1_i16(MIN);
            let mut grow_D_argmax = simd_set1_i16(0);
            let (D_max, D_argmax, right_max, down_max) = match dir {
                Direction::Right => {
                    off += (D_col.get(0) as i32) - (ZERO as i32);
                    #[cfg(feature = "debug")]
                    println!("off: {}", off);
                    let off_add = simd_set1_i16(clamp(prev_off - off));

                    if TRACE {
                        self.trace.add_block(self.i, self.j + block_size - step, step, block_size, true);
                    }

                    self.just_offset(block_size, D_col.as_mut_ptr(), C_col.as_mut_ptr(), off_add);

                    let (D_max, D_argmax) = self.place_block(
                        self.query,
                        self.reference,
                        self.i,
                        self.j + block_size - step,
                        step,
                        block_size,
                        D_col.as_mut_ptr(),
                        C_col.as_mut_ptr(),
                        temp_buf1.as_mut_ptr(),
                        temp_buf2.as_mut_ptr(),
                        true
                    );

                    let right_max = self.max(block_size, D_col.as_ptr());

                    // shift and offset bottom row
                    let down_max = self.shift_and_offset(
                        block_size,
                        D_row.as_mut_ptr(),
                        R_row.as_mut_ptr(),
                        temp_buf1.as_mut_ptr(),
                        temp_buf2.as_mut_ptr(),
                        off_add,
                        step
                    );

                    (D_max, D_argmax, right_max, down_max)
                },
                Direction::Down => {
                    off += (D_row.get(0) as i32) - (ZERO as i32);
                    #[cfg(feature = "debug")]
                    println!("off: {}", off);
                    let off_add = simd_set1_i16(clamp(prev_off - off));

                    if TRACE {
                        self.trace.add_block(self.i + block_size - step, self.j, block_size, step, false);
                    }

                    self.just_offset(block_size, D_row.as_mut_ptr(), R_row.as_mut_ptr(), off_add);

                    let (D_max, D_argmax) = self.place_block(
                        self.reference,
                        self.query,
                        self.j,
                        self.i + block_size - step,
                        step,
                        block_size,
                        D_row.as_mut_ptr(),
                        R_row.as_mut_ptr(),
                        temp_buf1.as_mut_ptr(),
                        temp_buf2.as_mut_ptr(),
                        false
                    );

                    let down_max = self.max(block_size, D_row.as_ptr());

                    // shift and offset last column
                    let right_max = self.shift_and_offset(
                        block_size,
                        D_col.as_mut_ptr(),
                        C_col.as_mut_ptr(),
                        temp_buf1.as_mut_ptr(),
                        temp_buf2.as_mut_ptr(),
                        off_add,
                        step
                    );

                    (D_max, D_argmax, right_max, down_max)
                },
                Direction::Grow => {
                    let grow_step = block_size - prev_size;

                    #[cfg(feature = "debug")]
                    println!("off: {}", off);
                    #[cfg(feature = "debug")]
                    println!("Grow down");

                    if TRACE {
                        self.trace.add_block(self.i + prev_size, self.j, prev_size, grow_step, false);
                    }

                    // down
                    let (D_max1, D_argmax1) = self.place_block(
                        self.reference,
                        self.query,
                        self.j,
                        self.i + prev_size,
                        grow_step,
                        prev_size,
                        D_row.as_mut_ptr(),
                        R_row.as_mut_ptr(),
                        D_col.as_mut_ptr().add(prev_size),
                        C_col.as_mut_ptr().add(prev_size),
                        false
                    );

                    #[cfg(feature = "debug")]
                    println!("Grow right");

                    if TRACE {
                        self.trace.add_block(self.i, self.j + prev_size, grow_step, block_size, true);
                    }

                    // right
                    let (D_max2, D_argmax2) = self.place_block(
                        self.query,
                        self.reference,
                        self.i,
                        self.j + prev_size,
                        grow_step,
                        block_size,
                        D_col.as_mut_ptr(),
                        C_col.as_mut_ptr(),
                        D_row.as_mut_ptr().add(prev_size),
                        R_row.as_mut_ptr().add(prev_size),
                        true
                    );

                    let right_max = self.max(block_size, D_col.as_ptr());
                    let down_max = self.max(block_size, D_row.as_ptr());
                    grow_D_max = D_max1;
                    grow_D_argmax = D_argmax1;

                    D_col_ckpt2.set_all(&D_col, block_size);
                    C_col_ckpt2.set_all(&C_col, block_size);
                    D_row_ckpt2.set_all(&D_row, block_size);
                    R_row_ckpt2.set_all(&R_row, block_size);

                    D_col_ckpt.set_all(&D_col, block_size);
                    C_col_ckpt.set_all(&C_col, block_size);
                    D_row_ckpt.set_all(&D_row, block_size);
                    R_row_ckpt.set_all(&R_row, block_size);
                    if TRACE {
                        self.trace.save_ckpt(true);
                    }

                    (D_max2, D_argmax2, right_max, down_max)
                }
            };

            let D_max_max = simd_hmax_i16(D_max);
            let grow_max = simd_hmax_i16(grow_D_max);
            let max = cmp::max(D_max_max, grow_max);
            let off_max = off + (max as i32) - (ZERO as i32);
            #[cfg(feature = "debug")]
            println!("down max: {}, right max: {}", down_max, right_max);

            y_drop_iter += 1;
            let mut grow_no_max = dir == Direction::Grow;

            if off_max > best_max {
                if X_DROP {
                    let lane_idx = simd_hargmax_i16(D_max, D_max_max);
                    let idx = simd_slow_extract_i16(D_argmax, lane_idx) as usize;
                    let r = (idx % (block_size / L)) * L + lane_idx;
                    let c = (block_size - step) + (idx / (block_size / L));

                    match dir {
                        Direction::Right => {
                            best_argmax_i = self.i + r;
                            best_argmax_j = self.j + c;
                        },
                        Direction::Down => {
                            best_argmax_i = self.i + c;
                            best_argmax_j = self.j + r;
                        },
                        Direction::Grow => {
                            if max >= grow_max {
                                best_argmax_i = self.i + (idx % (block_size / L)) * L + lane_idx;
                                best_argmax_j = self.j + prev_size + (idx / (block_size / L));
                            } else {
                                let lane_idx = simd_hargmax_i16(grow_D_max, grow_max);
                                let idx = simd_slow_extract_i16(grow_D_argmax, lane_idx) as usize;
                                best_argmax_i = self.i + prev_size + (idx / (prev_size / L));
                                best_argmax_j = self.j + (idx % (prev_size / L)) * L + lane_idx;
                            }
                        }
                    }
                }

                if block_size < self.max_size {
                    i_ckpt2 = i_ckpt;
                    j_ckpt2 = j_ckpt;
                    off_ckpt2 = off_ckpt;
                    D_col_ckpt2.set_all(&D_col_ckpt, block_size);
                    C_col_ckpt2.set_all(&C_col_ckpt, block_size);
                    D_row_ckpt2.set_all(&D_row_ckpt, block_size);
                    R_row_ckpt2.set_all(&R_row_ckpt, block_size);

                    i_ckpt = self.i;
                    j_ckpt = self.j;
                    off_ckpt = off;
                    D_col_ckpt.set_all(&D_col, block_size);
                    C_col_ckpt.set_all(&C_col, block_size);
                    D_row_ckpt.set_all(&D_row, block_size);
                    R_row_ckpt.set_all(&R_row, block_size);
                    if TRACE {
                        self.trace.save_ckpt(false);
                    }

                    grow_no_max = false;
                }

                best_max = off_max;
                y_drop_iter = 0;
            }

            if X_DROP && unlikely(off_max < best_max - self.x_drop) {
                // x drop termination
                break;
            }

            if unlikely(self.i + block_size > self.query.len() && self.j + block_size > self.reference.len()) {
                // reached the end of the strings
                break;
            }

            // first check if the shift direction is "forced" to avoid going out of bounds
            if unlikely(self.j + block_size > self.reference.len()) {
                self.i += step;
                dir = Direction::Down;
                continue;
            }
            if unlikely(self.i + block_size > self.query.len()) {
                self.j += step;
                dir = Direction::Right;
                continue;
            }

            let next_size = if GROW_EXP { block_size * 2 } else { block_size + GROW_STEP };
            if next_size <= self.max_size {
                if unlikely(y_drop_iter > (block_size / step) - 1/* / 2*/ || grow_no_max) {
                    // y drop grow block
                    prev_size = block_size;
                    block_size = next_size;
                    dir = Direction::Grow;
                    if STEP != LARGE_STEP && block_size >= (LARGE_STEP / STEP) * self.min_size {
                        step = LARGE_STEP;
                    }

                    self.i = i_ckpt2;
                    self.j = j_ckpt2;
                    off = off_ckpt2;
                    i_ckpt = i_ckpt2;
                    j_ckpt = j_ckpt2;
                    off_ckpt = off_ckpt2;
                    D_col.set_all(&D_col_ckpt2, prev_size);
                    C_col.set_all(&C_col_ckpt2, prev_size);
                    D_row.set_all(&D_row_ckpt2, prev_size);
                    R_row.set_all(&R_row_ckpt2, prev_size);
                    if TRACE {
                        self.trace.restore_ckpt();
                    }

                    y_drop_iter = 0;
                    continue;
                }
            }

            // move according to where the max is
            if down_max > right_max {
                self.i += step;
                dir = Direction::Down;
            } else {
                self.j += step;
                dir = Direction::Right;
            }
        }

        #[cfg(any(feature = "debug", feature = "debug_size"))]
        {
            println!("query size: {}, reference size: {}", self.query.len() - 1, self.reference.len() - 1);
            println!("end block size: {}", block_size);
        }

        self.res = if X_DROP {
            AlignResult {
                score: best_max,
                query_idx: best_argmax_i,
                reference_idx: best_argmax_j
            }
        } else {
            debug_assert!(self.i <= self.query.len());
            let score = off + match dir {
                Direction::Right | Direction::Grow => {
                    let idx = self.query.len() - self.i;
                    debug_assert!(idx < block_size);
                    (D_col.get(idx) as i32) - (ZERO as i32)
                },
                Direction::Down => {
                    let idx = self.reference.len() - self.j;
                    debug_assert!(idx < block_size);
                    (D_row.get(idx) as i32) - (ZERO as i32)
                }
            };
            AlignResult {
                score,
                query_idx: self.query.len(),
                reference_idx: self.reference.len()
            }
        };
    }

    #[allow(non_snake_case)]
    #[inline]
    unsafe fn just_offset(&self, block_size: usize, buf1: *mut i16, buf2: *mut i16, off_add: Simd) {
        let mut i = 0;
        while i < block_size {
            let a = simd_adds_i16(simd_load(buf1.add(i) as _), off_add);
            let b = simd_adds_i16(simd_load(buf2.add(i) as _), off_add);
            simd_store(buf1.add(i) as _, a);
            simd_store(buf2.add(i) as _, b);
            i += L;
        }
    }

    #[allow(non_snake_case)]
    #[inline]
    unsafe fn max(&self, block_size: usize, buf: *const i16) -> i16 {
        let mut curr_max = simd_set1_i16(MIN);
        let mut i = 0;
        while i < block_size {
            curr_max = simd_max_i16(curr_max, simd_load(buf.add(i) as _));
            i += L;
        }
        simd_hmax_i16(curr_max)
    }

    #[allow(non_snake_case)]
    #[inline]
    unsafe fn shift_and_offset(&self, block_size: usize, buf1: *mut i16, buf2: *mut i16, temp_buf1: *mut i16, temp_buf2: *mut i16, off_add: Simd, step: usize) -> i16 {
        #[inline]
        unsafe fn sr(a: Simd, b: Simd, step: usize) -> Simd {
            if STEP == LARGE_STEP {
                simd_sr_i16!(a, b, STEP)
            } else {
                if step == STEP {
                    simd_sr_i16!(a, b, STEP)
                } else {
                    simd_sr_i16!(a, b, LARGE_STEP)
                }
            }
        }
        let mut curr_max = simd_set1_i16(MIN);
        let mut curr1 = simd_adds_i16(simd_load(buf1 as _), off_add);
        let mut curr2 = simd_adds_i16(simd_load(buf2 as _), off_add);

        let mut i = 0;
        while i < block_size - L {
            let next1 = simd_adds_i16(simd_load(buf1.add(i + L) as _), off_add);
            let next2 = simd_adds_i16(simd_load(buf2.add(i + L) as _), off_add);
            let shifted = sr(next1, curr1, step);
            simd_store(buf1.add(i) as _, shifted);
            simd_store(buf2.add(i) as _, sr(next2, curr2, step));
            curr_max = simd_max_i16(curr_max, shifted);
            curr1 = next1;
            curr2 = next2;
            i += L;
        }

        let next1 = simd_load(temp_buf1 as _);
        let next2 = simd_load(temp_buf2 as _);
        let shifted = sr(next1, curr1, step);
        simd_store(buf1.add(block_size - L) as _, shifted);
        simd_store(buf2.add(block_size - L) as _, sr(next2, curr2, step));
        curr_max = simd_max_i16(curr_max, shifted);
        simd_hmax_i16(curr_max)
    }

    // Place block right or down.
    //
    // Assumes all inputs are already relative to the current offset.
    #[allow(non_snake_case)]
    #[inline]
    unsafe fn place_block(&mut self,
                          query: &PaddedBytes,
                          reference: &PaddedBytes,
                          start_i: usize,
                          start_j: usize,
                          width: usize,
                          height: usize,
                          D_col: *mut i16,
                          C_col: *mut i16,
                          D_row: *mut i16,
                          R_row: *mut i16,
                          right: bool) -> (Simd, Simd) {
        let (gap_open, gap_extend) = self.get_const_simd();
        let mut D_max = simd_set1_i16(MIN);
        let mut D_argmax = simd_set1_i16(0);
        let mut curr_i = simd_set1_i16(0);

        if unlikely(width == 0 || height == 0) {
            return (D_max, D_argmax);
        }

        // hottest loop in the whole program
        for j in 0..width {
            let mut D_corner = simd_set1_i16(MIN);
            let mut R_insert = simd_set1_i16(MIN);

            let c = reference.get(start_j + j);

            let mut i = 0;
            while i < height {
                #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "mca"))]
                asm!("# LLVM-MCA-BEGIN place_block inner loop", options(nomem, nostack, preserves_flags));

                let D10 = simd_load(D_col.add(i) as _);
                let C10 = simd_load(C_col.add(i) as _);
                let D00 = simd_sl_i16!(D10, D_corner, 1);
                D_corner = D10;

                let scores = self.matrix.get_scores(c, halfsimd_loadu(query.as_ptr(start_i + i) as _), right);
                let mut D11 = simd_adds_i16(D00, scores);
                if unlikely(start_i + i == 0 && start_j + j == 0) {
                    D11 = simd_insert_i16!(D11, ZERO, 0);
                }

                let C11 = simd_max_i16(simd_adds_i16(C10, gap_extend), simd_adds_i16(D10, gap_open));
                D11 = simd_max_i16(D11, C11);

                let D11_open = simd_adds_i16(D11, gap_open);
                let mut R11 = simd_sl_i16!(D11_open, R_insert, 1);
                R11 = simd_prefix_scan_i16(R11, self.gaps.extend as i16);
                D11 = simd_max_i16(D11, R11);
                R_insert = simd_max_i16(D11_open, simd_adds_i16(R11, gap_extend));

                #[cfg(feature = "debug")]
                {
                    print!("s:   ");
                    simd_dbg_i16(scores);
                    print!("D00: ");
                    simd_dbg_i16(D00);
                    print!("C11: ");
                    simd_dbg_i16(C11);
                    print!("R11: ");
                    simd_dbg_i16(R11);
                    print!("D11: ");
                    simd_dbg_i16(D11);
                }

                if TRACE {
                    let trace_D_C = simd_cmpeq_i16(D11, C11);
                    let trace_D_R = simd_cmpeq_i16(D11, R11);
                    #[cfg(feature = "debug")]
                    {
                        print!("D_C: ");
                        simd_dbg_i16(trace_D_C);
                        print!("D_R: ");
                        simd_dbg_i16(trace_D_R);
                    }
                    let trace = simd_movemask_i8(simd_packus_i16(trace_D_C, trace_D_R));
                    self.trace.add_trace(trace);
                }

                D_max = simd_max_i16(D_max, D11);

                if X_DROP {
                    let mask = simd_cmpeq_i16(D_max, D11);
                    D_argmax = simd_blend_i8(D_argmax, curr_i, mask);
                    curr_i = simd_adds_i16(curr_i, simd_set1_i16(1));
                }

                simd_store(D_col.add(i) as _, D11);
                simd_store(C_col.add(i) as _, C11);
                i += L;

                #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "mca"))]
                asm!("# LLVM-MCA-END", options(nomem, nostack, preserves_flags));
            }

            ptr::write(D_row.add(j), *D_col.add(height - 1));
            // must subtract gap_extend from R_insert due to how R_insert is calculated
            ptr::write(R_row.add(j), simd_extract_i16!(simd_subs_i16(R_insert, gap_extend), L - 1));

            if !X_DROP && unlikely(start_i + height > query.len()
                                   && start_j + j >= reference.len()) {
                break;
            }
        }

        (D_max, D_argmax)
    }

    #[inline]
    pub fn res(&self) -> AlignResult {
        self.res
    }

    #[inline]
    pub fn trace(&self) -> &Trace {
        assert!(TRACE);
        &self.trace
    }

    #[inline]
    unsafe fn get_const_simd(&self) -> (Simd, Simd) {
        // some useful constant simd vectors
        let gap_open = simd_set1_i16(self.gaps.open as i16);
        let gap_extend = simd_set1_i16(self.gaps.extend as i16);
        (gap_open, gap_extend)
    }
}

#[derive(Clone)]
pub struct Trace {
    trace: Vec<TraceType>,
    right: Vec<u64>,
    block_start: Vec<u32>,
    block_size: Vec<u32>,
    trace_idx: usize,
    block_idx: usize,
    ckpt_trace_idx: usize,
    ckpt_block_idx: usize,
    ckpt_trace_idx2: usize,
    ckpt_block_idx2: usize
}

impl Trace {
    #[inline]
    pub fn new(query_len: usize, reference_len: usize, block_size: usize) -> Self {
        let len = query_len + reference_len;
        let trace = Vec::with_capacity(len * (block_size / L));
        let right = vec![0u64; div_ceil(len, 64)];
        let block_start = vec![0u32; len * 2];
        let block_size = vec![0u32; len * 2];

        Self {
            trace,
            right,
            block_start,
            block_size,
            trace_idx: 0,
            block_idx: 0,
            ckpt_trace_idx: 0,
            ckpt_block_idx: 0,
            ckpt_trace_idx2: 0,
            ckpt_block_idx2: 0
        }
    }

    #[inline]
    pub fn add_trace(&mut self, t: TraceType) {
        debug_assert!(self.trace_idx < self.trace.len());
        unsafe { *self.trace.get_unchecked_mut(self.trace_idx) = t; }
        self.trace_idx += 1;
    }

    #[inline]
    pub fn add_block(&mut self, i: usize, j: usize, width: usize, height: usize, right: bool) {
        debug_assert!(self.block_idx * 2 < self.block_start.len());
        unsafe {
            *self.block_start.get_unchecked_mut(self.block_idx * 2) = i as u32;
            *self.block_start.get_unchecked_mut(self.block_idx * 2 + 1) = j as u32;
            *self.block_size.get_unchecked_mut(self.block_idx * 2) = height as u32;
            *self.block_size.get_unchecked_mut(self.block_idx * 2 + 1) = width as u32;

            let a = self.block_idx / 64;
            let b = self.block_idx % 64;
            let v = *self.right.get_unchecked(a) & !(1 << b); // clear bit
            *self.right.get_unchecked_mut(a) = v | ((right as u64) << b);

            self.trace.resize(self.trace.len() + width * height / L, 0 as TraceType);

            self.block_idx += 1;
        }
    }

    #[inline]
    pub fn save_ckpt(&mut self, set2: bool) {
        if set2 {
            self.ckpt_trace_idx2 = self.trace.len();
            self.ckpt_block_idx2 = self.block_idx;
        } else {
            self.ckpt_trace_idx2 = self.ckpt_trace_idx;
            self.ckpt_block_idx2 = self.ckpt_block_idx;
        }
        self.ckpt_trace_idx = self.trace.len();
        self.ckpt_block_idx = self.block_idx;
    }

    #[inline]
    pub fn restore_ckpt(&mut self) {
        unsafe { self.trace.set_len(self.ckpt_trace_idx2); }
        self.trace_idx = self.ckpt_trace_idx2;
        self.block_idx = self.ckpt_block_idx2;
        self.ckpt_trace_idx = self.ckpt_trace_idx2;
        self.ckpt_block_idx = self.ckpt_block_idx2;
    }

    pub fn cigar(&self, mut i: usize, mut j: usize) -> Cigar {
        unsafe {
            let mut res = Cigar::new(i + j + 5);
            let mut block_idx = self.block_idx;
            let mut trace_idx = self.trace.len();
            let mut block_i;
            let mut block_j;
            let mut block_width;
            let mut block_height;
            let mut right;

            // use lookup table instead of hard to predict branches
            static OP_LUT: [(Operation, usize, usize); 8] = [
                (Operation::M, 1, 1), // 0b000
                (Operation::I, 1, 0), // 0b001
                (Operation::D, 0, 1), // 0b010
                (Operation::I, 1, 0), // 0b011, bias towards i -= 1 to avoid going out of bounds
                (Operation::M, 1, 1), // 0b100
                (Operation::D, 0, 1), // 0b101
                (Operation::I, 1, 0), // 0b110
                (Operation::D, 0, 1) // 0b111, bias towards j -= 1 to avoid going out of bounds
            ];

            while i > 0 || j > 0 {
                loop {
                    block_idx -= 1;
                    block_i = *self.block_start.get_unchecked(block_idx * 2) as usize;
                    block_j = *self.block_start.get_unchecked(block_idx * 2 + 1) as usize;
                    block_height = *self.block_size.get_unchecked(block_idx * 2) as usize;
                    block_width = *self.block_size.get_unchecked(block_idx * 2 + 1) as usize;
                    trace_idx -= block_width * block_height / L;

                    if i >= block_i && j >= block_j {
                        right = (((*self.right.get_unchecked(block_idx / 64) >> (block_idx % 64)) & 0b1) << 2) as usize;
                        break;
                    }
                }

                if right > 0 {
                    while i >= block_i && j >= block_j && (i > 0 || j > 0) {
                        let curr_i = i - block_i;
                        let curr_j = j - block_j;
                        let idx = trace_idx + curr_i / L + curr_j * (block_height / L);
                        let t = ((*self.trace.get_unchecked(idx) >> ((curr_i % L) * 2)) & 0b11) as usize;
                        let lut_idx = right | t;
                        let op = OP_LUT[lut_idx].0;
                        i -= OP_LUT[lut_idx].1;
                        j -= OP_LUT[lut_idx].2;
                        res.add(op);
                    }
                } else {
                    while i >= block_i && j >= block_j && (i > 0 || j > 0) {
                        let curr_i = i - block_i;
                        let curr_j = j - block_j;
                        let idx = trace_idx + curr_j / L + curr_i * (block_width / L);
                        let t = ((*self.trace.get_unchecked(idx) >> ((curr_j % L) * 2)) & 0b11) as usize;
                        let lut_idx = right | t;
                        let op = OP_LUT[lut_idx].0;
                        i -= OP_LUT[lut_idx].1;
                        j -= OP_LUT[lut_idx].2;
                        res.add(op);
                    }
                }
            }

            res
        }
    }
}

#[inline]
fn clamp(x: i32) -> i16 {
    cmp::min(cmp::max(x, MIN as i32), i16::MAX as i32) as i16
}

#[inline]
fn div_ceil(n: usize, d: usize) -> usize {
    (n + d - 1) / d
}

pub struct Aligned {
    layout: alloc::Layout,
    ptr: *const i16
}

impl Aligned {
    pub unsafe fn new(block_size: usize) -> Self {
        let layout = alloc::Layout::from_size_align_unchecked(block_size * 2, L_BYTES);
        let ptr = alloc::alloc_zeroed(layout) as *const i16;
        let mut i = 0;
        while i < block_size {
            simd_store(ptr.add(i) as _, simd_set1_i16(MIN));
            i += L;
        }
        Self { layout, ptr }
    }

    pub unsafe fn set_all(&mut self, o: &Aligned, len: usize) {
        let o_ptr = o.as_ptr();
        let mut i = 0;
        while i < len {
            simd_store(self.ptr.add(i) as _, simd_load(o_ptr.add(i) as _));
            i += L;
        }
    }

    #[inline]
    pub fn get(&self, i: usize) -> i16 {
        unsafe { *self.ptr.add(i) }
    }

    #[inline]
    pub fn set(&mut self, i: usize, v: i16) {
        unsafe { ptr::write(self.ptr.add(i) as _, v); }
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut i16 {
        self.ptr as _
    }

    #[inline]
    pub fn as_ptr(&self) -> *const i16 {
        self.ptr
    }
}

impl Drop for Aligned {
    fn drop(&mut self) {
        unsafe { alloc::dealloc(self.ptr as _, self.layout); }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct PaddedBytes {
    s: Vec<u8>,
    len: usize
}

impl PaddedBytes {
    #[inline]
    pub fn from_bytes<M: Matrix>(b: &[u8], block_size: usize, matrix: &M) -> Self {
        let mut v = b.to_owned();
        let len = v.len();
        v.insert(0, NULL);
        v.resize(v.len() + block_size, NULL);
        v.iter_mut().for_each(|c| *c = matrix.convert_char(*c));
        Self { s: v, len }
    }

    #[inline]
    pub fn from_str<M: Matrix>(s: &str, block_size: usize, matrix: &M) -> Self {
        Self::from_bytes(s.as_bytes(), block_size, matrix)
    }

    #[inline]
    pub fn from_string<M: Matrix>(s: String, block_size: usize, matrix: &M) -> Self {
        let mut v = s.into_bytes();
        let len = v.len();
        v.insert(0, NULL);
        v.resize(v.len() + block_size, NULL);
        v.iter_mut().for_each(|c| *c = matrix.convert_char(*c));
        Self { s: v, len }
    }

    #[inline]
    pub fn get(&self, i: usize) -> u8 {
        unsafe { *self.s.get_unchecked(i) }
    }

    #[inline]
    pub fn set(&mut self, i: usize, c: u8) {
        unsafe { *self.s.get_unchecked_mut(i) = c; }
    }

    #[inline]
    pub fn as_ptr(&self, i: usize) -> *const u8 {
        unsafe { self.s.as_ptr().add(i) }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct AlignResult {
    pub score: i32,
    pub query_idx: usize,
    pub reference_idx: usize
}

#[derive(Copy, Clone, PartialEq, Debug)]
enum Direction {
    Right,
    Down,
    Grow
}

#[cfg(test)]
mod tests {
    use crate::scores::*;

    use super::*;

    #[test]
    fn test_no_x_drop() {
        let test_gaps = Gaps { open: -11, extend: -1 };

        let r = PaddedBytes::from_bytes(b"AAAA", 16, &BLOSUM62);
        let q = PaddedBytes::from_bytes(b"AARA", 16, &BLOSUM62);
        let a = Block::<_, false, false>::align(&q, &r, &BLOSUM62, test_gaps, 16..=16, 0);
        assert_eq!(a.res().score, 11);

        let r = PaddedBytes::from_bytes(b"AAAA", 16, &BLOSUM62);
        let q = PaddedBytes::from_bytes(b"AAAA", 16, &BLOSUM62);
        let a = Block::<_, false, false>::align(&q, &r, &BLOSUM62, test_gaps, 16..=16, 0);
        assert_eq!(a.res().score, 16);

        let r = PaddedBytes::from_bytes(b"AAAA", 16, &BLOSUM62);
        let q = PaddedBytes::from_bytes(b"AARA", 16, &BLOSUM62);
        let a = Block::<_, false, false>::align(&q, &r, &BLOSUM62, test_gaps, 16..=16, 0);
        assert_eq!(a.res().score, 11);

        let r = PaddedBytes::from_bytes(b"AAAA", 16, &BLOSUM62);
        let q = PaddedBytes::from_bytes(b"RRRR", 16, &BLOSUM62);
        let a = Block::<_, false, false>::align(&q, &r, &BLOSUM62, test_gaps, 16..=16, 0);
        assert_eq!(a.res().score, -4);

        let r = PaddedBytes::from_bytes(b"AAAA", 16, &BLOSUM62);
        let q = PaddedBytes::from_bytes(b"AAA", 16, &BLOSUM62);
        let a = Block::<_, false, false>::align(&q, &r, &BLOSUM62, test_gaps, 16..=16, 0);
        assert_eq!(a.res().score, 1);

        let test_gaps2 = Gaps { open: -1, extend: -1 };

        let r = PaddedBytes::from_bytes(b"AAAN", 16, &NW1);
        let q = PaddedBytes::from_bytes(b"ATAA", 16, &NW1);
        let a = Block::<_, false, false>::align(&q, &r, &NW1, test_gaps2, 16..=16, 0);
        assert_eq!(a.res().score, 1);

        let r = PaddedBytes::from_bytes(b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", 16, &NW1);
        let q = PaddedBytes::from_bytes(b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", 16, &NW1);
        let a = Block::<_, false, false>::align(&q, &r, &NW1, test_gaps2, 16..=16, 0);
        assert_eq!(a.res().score, 32);

        let r = PaddedBytes::from_bytes(b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", 16, &NW1);
        let q = PaddedBytes::from_bytes(b"TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT", 16, &NW1);
        let a = Block::<_, false, false>::align(&q, &r, &NW1, test_gaps2, 16..=16, 0);
        assert_eq!(a.res().score, -32);

        let r = PaddedBytes::from_bytes(b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", 16, &NW1);
        let q = PaddedBytes::from_bytes(b"TATATATATATATATATATATATATATATATA", 16, &NW1);
        let a = Block::<_, false, false>::align(&q, &r, &NW1, test_gaps2, 16..=16, 0);
        assert_eq!(a.res().score, 0);

        let r = PaddedBytes::from_bytes(b"TTAAAAAAATTTTTTTTTTTT", 16, &NW1);
        let q = PaddedBytes::from_bytes(b"TTTTTTTTAAAAAAATTTTTTTTT", 16, &NW1);
        let a = Block::<_, false, false>::align(&q, &r, &NW1, test_gaps2, 16..=16, 0);
        assert_eq!(a.res().score, 9);

        let r = PaddedBytes::from_bytes(b"AAAA", 16, &NW1);
        let q = PaddedBytes::from_bytes(b"C", 16, &NW1);
        let a = Block::<_, false, false>::align(&q, &r, &NW1, test_gaps2, 16..=16, 0);
        assert_eq!(a.res().score, -4);
        let a = Block::<_, false, false>::align(&r, &q, &NW1, test_gaps2, 16..=16, 0);
        assert_eq!(a.res().score, -4);
    }

    #[test]
    fn test_x_drop() {
        let test_gaps = Gaps { open: -11, extend: -1 };

        let r = PaddedBytes::from_bytes(b"AAARRA", 16, &BLOSUM62);
        let q = PaddedBytes::from_bytes(b"AAAAAA", 16, &BLOSUM62);
        let a = Block::<_, false, true>::align(&q, &r, &BLOSUM62, test_gaps, 16..=16, 1);
        assert_eq!(a.res(), AlignResult { score: 14, query_idx: 6, reference_idx: 6 });

        let r = PaddedBytes::from_bytes(b"AAAAAAAAAAAAAAARRRRRRRRRRRRRRRRAAAAAAAAAAAAA", 16, &BLOSUM62);
        let q = PaddedBytes::from_bytes(b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", 16, &BLOSUM62);
        let a = Block::<_, false, true>::align(&q, &r, &BLOSUM62, test_gaps, 16..=16, 1);
        assert_eq!(a.res(), AlignResult { score: 60, query_idx: 15, reference_idx: 15 });
    }

    #[test]
    fn test_trace() {
        let test_gaps = Gaps { open: -11, extend: -1 };

        let r = PaddedBytes::from_bytes(b"AAARRA", 16, &BLOSUM62);
        let q = PaddedBytes::from_bytes(b"AAAAAA", 16, &BLOSUM62);
        let a = Block::<_, true, false>::align(&q, &r, &BLOSUM62, test_gaps, 16..=16, 0);
        let res = a.res();
        assert_eq!(res, AlignResult { score: 14, query_idx: 6, reference_idx: 6 });
        assert_eq!(a.trace().cigar(res.query_idx, res.reference_idx).to_string(), "6M");

        let r = PaddedBytes::from_bytes(b"AAAA", 16, &BLOSUM62);
        let q = PaddedBytes::from_bytes(b"AAA", 16, &BLOSUM62);
        let a = Block::<_, true, false>::align(&q, &r, &BLOSUM62, test_gaps, 16..=16, 0);
        let res = a.res();
        assert_eq!(res, AlignResult { score: 1, query_idx: 3, reference_idx: 4 });
        assert_eq!(a.trace().cigar(res.query_idx, res.reference_idx).to_string(), "3M1D");

        let test_gaps2 = Gaps { open: -1, extend: -1 };

        let r = PaddedBytes::from_bytes(b"TTAAAAAAATTTTTTTTTTTT", 16, &NW1);
        let q = PaddedBytes::from_bytes(b"TTTTTTTTAAAAAAATTTTTTTTT", 16, &NW1);
        let a = Block::<_, true, false>::align(&q, &r, &NW1, test_gaps2, 16..=16, 0);
        let res = a.res();
        assert_eq!(res, AlignResult { score: 9, query_idx: 24, reference_idx: 21 });
        assert_eq!(a.trace().cigar(res.query_idx, res.reference_idx).to_string(), "2M6I16M3D");
    }
}
