#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::avx2::*;

#[cfg(target_arch = "wasm32")]
use crate::simd128::*;

use crate::scores::*;

use std::{cmp, ptr, i16, alloc};
use std::marker::PhantomData;

const NULL: u8 = b'A' + 26u8; // this null byte value works for both amino acids and nucleotides

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

// TODO: create matrices with const fn

pub struct Block<'a, P: ScoreParams, M: 'a + Matrix, const MIN_SIZE: usize, const MAX_SIZE: usize, const TRACE: bool, const X_DROP: bool> {
    res: AlignResult,
    trace: Trace,
    query: &'a PaddedBytes,
    i: usize,
    reference: &'a PaddedBytes,
    j: usize,
    matrix: &'a M,
    x_drop: i32,
    y_drop: i32,
    grow_y_drop: i32,
    _phantom: PhantomData<P>
}

impl<'a, P: ScoreParams, M: 'a + Matrix, const MIN_SIZE: usize, const MAX_SIZE: usize, const TRACE: bool, const X_DROP: bool> Block<'a, P, M, { MIN_SIZE }, { MAX_SIZE }, { TRACE }, { X_DROP }> {
    const STEP: usize = L / 2;
    const EVEN_BITS: u32 = 0x55555555u32;

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
    pub fn align(query: &'a PaddedBytes, reference: &'a PaddedBytes, matrix: &'a M, x_drop: i32, y_drop: i32, grow_y_drop: i32) -> Self {
        assert!(P::GAP_OPEN <= P::GAP_EXTEND);
        assert!(y_drop >= 0);
        assert!(grow_y_drop >= 0);
        assert!(MIN_SIZE >= L);

        if X_DROP {
            assert!(x_drop >= 0);
        }

        let mut a = Self {
            res: AlignResult { score: 0, query_idx: 0, reference_idx: 0 },
            trace: if TRACE { Trace::new(query.len(), reference.len()) } else { Trace::new(0, 0) },
            query,
            i: 0,
            reference,
            j: 0,
            matrix,
            x_drop,
            y_drop,
            grow_y_drop,
            _phantom: PhantomData
        };

        unsafe { a.align_core(); }
        a
    }

    #[cfg_attr(any(target_arch = "x86", target_arch = "x86_64"), target_feature(enable = "avx2"))]
    #[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
    #[allow(non_snake_case)]
    unsafe fn align_core(&mut self) {
        let mut best_max = 0i32;
        let mut best_argmax_i = 0usize;
        let mut best_argmax_j = 0usize;

        let mut dir = Direction::Grow;
        let mut block_size = L;

        let mut off = 0i32;
        let mut prev_off;

        let mut D_col = Aligned::new(MAX_SIZE);
        let mut C_col = Aligned::new(MAX_SIZE);
        let mut D_row = Aligned::new(MAX_SIZE);
        let mut R_row = Aligned::new(MAX_SIZE);
        for i in 0..MAX_SIZE {
            let D_insert = if i == 0 {
                0
            } else {
                (P::GAP_OPEN as i16) + ((i - 1) as i16) * (P::GAP_EXTEND as i16)
            };
            D_col.set(i, D_insert);
        }
        self.j += 1;

        let mut temp_buf1 = Aligned::new(L);
        let mut temp_buf2 = Aligned::new(L);

        let mut i_ckpt = self.i;
        let mut j_ckpt = self.j;
        let mut off_ckpt = 0i32;
        let mut D_col_ckpt = Aligned::new(MAX_SIZE);
        let mut C_col_ckpt = Aligned::new(MAX_SIZE);
        let mut D_row_ckpt = Aligned::new(MAX_SIZE);
        let mut R_row_ckpt = Aligned::new(MAX_SIZE);
        D_col_ckpt.set_all(&D_col);

        loop {
            #[cfg(feature = "debug")]
            {
                println!("i: {}", self.i);
                println!("j: {}", self.j);
                println!("{:?}", dir);
                println!("off: {}", off);
                println!("block size: {}", block_size);
            }

            prev_off = off;
            let mut grow_D_max = simd_set1_i16(i16::MIN);
            let mut grow_D_argmax = simd_set1_i16(0);
            let (D_max, D_argmax, right_max, down_max) = match dir {
                Direction::Right => {
                    off += D_col.get(0) as i32;
                    let off_add = simd_set1_i16(clamp(prev_off - off));

                    // offset previous column
                    self.just_offset(block_size, D_col.as_mut_ptr(), C_col.as_mut_ptr(), off_add);

                    let (D_max, D_argmax, right_max) = self.place_block(
                        self.query,
                        self.reference,
                        self.i,
                        self.j + block_size - Self::STEP,
                        Self::STEP,
                        block_size,
                        D_col.as_mut_ptr(),
                        C_col.as_mut_ptr(),
                        temp_buf1.as_mut_ptr(),
                        temp_buf2.as_mut_ptr()
                    );

                    // shift and offset bottom row
                    let down_max = self.shift_and_offset(
                        block_size,
                        D_row.as_mut_ptr(),
                        R_row.as_mut_ptr(),
                        temp_buf1.as_mut_ptr(),
                        temp_buf2.as_mut_ptr(),
                        off_add
                    );

                    (D_max, D_argmax, right_max, down_max)
                },
                Direction::Down => {
                    off += D_row.get(0) as i32;
                    let off_add = simd_set1_i16(clamp(prev_off - off));

                    // offset previous row
                    self.just_offset(block_size, D_row.as_mut_ptr(), R_row.as_mut_ptr(), off_add);

                    let (D_max, D_argmax, down_max) = self.place_block(
                        self.reference,
                        self.query,
                        self.j,
                        self.i + block_size - Self::STEP,
                        Self::STEP,
                        block_size,
                        D_row.as_mut_ptr(),
                        R_row.as_mut_ptr(),
                        temp_buf1.as_mut_ptr(),
                        temp_buf2.as_mut_ptr()
                    );

                    // shift and offset last column
                    let right_max = self.shift_and_offset(
                        block_size,
                        D_col.as_mut_ptr(),
                        C_col.as_mut_ptr(),
                        temp_buf1.as_mut_ptr(),
                        temp_buf2.as_mut_ptr(),
                        off_add
                    );

                    (D_max, D_argmax, right_max, down_max)
                },
                Direction::Grow => {
                    let prev_size = block_size - L;

                    #[cfg(feature = "debug")]
                    println!("Grow down");

                    // down
                    let (D_max1, D_argmax1, down_max) = self.place_block(
                        self.reference,
                        self.query,
                        self.j,
                        self.i + prev_size,
                        L,
                        prev_size,
                        D_row.as_mut_ptr(),
                        R_row.as_mut_ptr(),
                        D_col.as_mut_ptr().add(prev_size),
                        C_col.as_mut_ptr().add(prev_size)
                    );

                    #[cfg(feature = "debug")]
                    println!("Grow right");

                    // right
                    let (D_max2, D_argmax2, right_max) = self.place_block(
                        self.query,
                        self.reference,
                        self.i,
                        self.j + prev_size,
                        L,
                        block_size,
                        D_col.as_mut_ptr(),
                        C_col.as_mut_ptr(),
                        D_row.as_mut_ptr().add(prev_size),
                        R_row.as_mut_ptr().add(prev_size)
                    );
                    let new_down_max = simd_hmax_i16(simd_load(D_row.as_ptr().add(prev_size) as _));
                    grow_D_max = D_max1;
                    grow_D_argmax = D_argmax1;

                    D_col_ckpt.set_all(&D_col);
                    C_col_ckpt.set_all(&C_col);
                    D_row_ckpt.set_all(&D_row);
                    R_row_ckpt.set_all(&R_row);

                    (D_max2, D_argmax2, right_max, cmp::max(down_max, new_down_max))
                }
            };

            let D_max_max = simd_hmax_i16(D_max);
            let grow_max = simd_hmax_i16(grow_D_max);
            let max = cmp::max(D_max_max, grow_max);
            let edge_max = off + cmp::max(right_max, down_max) as i32;

            if off + (max as i32) > best_max {
                if X_DROP {
                    let lane_idx = simd_hargmax_i16(D_max, D_max_max);
                    let idx = simd_slow_extract_i16(D_argmax, lane_idx) as usize;
                    let r = (idx % (block_size / L)) * L + lane_idx;
                    let c = (block_size - Self::STEP) + (idx / (block_size / L));

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
                            let prev_size = block_size - L;
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

                i_ckpt = self.i;
                j_ckpt = self.j;
                off_ckpt = off;
                D_col_ckpt.set_all(&D_col);
                C_col_ckpt.set_all(&C_col);
                D_row_ckpt.set_all(&D_row);
                R_row_ckpt.set_all(&R_row);

                best_max = off + max as i32;
            }

            if X_DROP && edge_max < best_max - self.x_drop {
                // x drop termination
                break;
            }

            if self.i + block_size > self.query.len() && self.j + block_size > self.reference.len() {
                // reached the end of the strings
                break;
            }

            // first check if the shift direction is "forced" to avoid going out of bounds
            if self.j + block_size > self.reference.len() {
                self.i += Self::STEP;
                dir = Direction::Down;
                continue;
            }
            if self.i + block_size > self.query.len() {
                self.j += Self::STEP;
                dir = Direction::Right;
                continue;
            }

            if block_size < MAX_SIZE && (block_size < MIN_SIZE || edge_max < best_max - self.y_drop) {
                // y drop grow block
                block_size += L;
                self.y_drop += self.grow_y_drop;
                dir = Direction::Grow;

                self.i = i_ckpt;
                self.j = j_ckpt;
                off = off_ckpt;
                D_col.set_all(&D_col_ckpt);
                C_col.set_all(&C_col_ckpt);
                D_row.set_all(&D_row_ckpt);
                R_row.set_all(&R_row_ckpt);

                continue;
            }

            // move according to where the max is
            if down_max > right_max {
                self.i += Self::STEP;
                dir = Direction::Down;
            } else {
                self.j += Self::STEP;
                dir = Direction::Right;
            }
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
                    D_col.get(idx) as i32
                },
                Direction::Down => {
                    let idx = self.reference.len() - self.j;
                    debug_assert!(idx < block_size);
                    D_row.get(idx) as i32
                }
            };
            AlignResult {
                score,
                query_idx: self.query.len(),
                reference_idx: self.reference.len()
            }
        };
    }

    #[cfg_attr(any(target_arch = "x86", target_arch = "x86_64"), target_feature(enable = "avx2"))]
    #[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
    #[allow(non_snake_case)]
    #[inline]
    unsafe fn just_offset(&self, block_size: usize, buf1: *mut i16, buf2: *mut i16, off_add: Simd) {
        for i in (0..block_size).step_by(L) {
            let curr1 = simd_adds_i16(simd_load(buf1.add(i) as _), off_add);
            let curr2 = simd_adds_i16(simd_load(buf2.add(i) as _), off_add);
            simd_store(buf1.add(i) as _, curr1);
            simd_store(buf2.add(i) as _, curr2);
        }
    }

    #[cfg_attr(any(target_arch = "x86", target_arch = "x86_64"), target_feature(enable = "avx2"))]
    #[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
    #[allow(non_snake_case)]
    #[inline]
    unsafe fn shift_and_offset(&self, block_size: usize, buf1: *mut i16, buf2: *mut i16, temp_buf1: *mut i16, temp_buf2: *mut i16, off_add: Simd) -> i16 {
        let neg_inf = simd_set1_i16(i16::MIN);
        let mut curr_max = neg_inf;
        let mut curr1 = simd_adds_i16(simd_load(buf1 as _), off_add);
        let mut curr2 = simd_adds_i16(simd_load(buf2 as _), off_add);

        for i in (0..block_size - L).step_by(L) {
            let next1 = simd_adds_i16(simd_load(buf1.add(i + L) as _), off_add);
            let next2 = simd_adds_i16(simd_load(buf2.add(i + L) as _), off_add);
            simd_store(buf1.add(i) as _, simd_sr_i16!(next1, curr1, Self::STEP));
            simd_store(buf2.add(i) as _, simd_sr_i16!(next2, curr2, Self::STEP));
            curr_max = simd_max_i16(curr_max, next1);
            curr1 = next1;
            curr2 = next2;
        }

        let next1 = simd_load(temp_buf1 as _);
        let next2 = simd_load(temp_buf2 as _);
        simd_store(buf1.add(block_size - L) as _, simd_sr_i16!(next1, curr1, Self::STEP));
        simd_store(buf2.add(block_size - L) as _, simd_sr_i16!(next2, curr2, Self::STEP));
        simd_hmax_i16(simd_max_i16(curr_max, next1))
    }

    // Place block right or down.
    //
    // Assumes all inputs are already relative to the current offset.
    #[cfg_attr(any(target_arch = "x86", target_arch = "x86_64"), target_feature(enable = "avx2"))]
    #[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
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
                          R_row: *mut i16) -> (Simd, Simd, i16) {
        let (neg_inf, gap_open, gap_extend) = self.get_const_simd();
        let mut D_max = neg_inf;
        let mut right_max = neg_inf;
        let mut D_argmax = simd_set1_i16(0);
        let mut curr_i = simd_set1_i16(0);

        if width == 0 || height == 0 {
            return (D_max, D_argmax, i16::MIN);
        }

        // TODO: trace direction

        // hottest loop in the whole program
        for j in 0..width {
            let mut D_corner = neg_inf;
            let mut R_insert = neg_inf;
            right_max = neg_inf;

            // efficiently lookup scores for each query character
            let matrix_ptr = self.matrix.as_ptr(reference.get(start_j + j) as usize);
            let scores1 = halfsimd_load(matrix_ptr as *const HalfSimd);
            let scores2 = if M::NUC {
                halfsimd_set1_i8(0) // unused, should be optimized out
            } else {
                halfsimd_load((matrix_ptr as *const HalfSimd).add(1))
            };

            for i in (0..height).step_by(L) {
                let D10 = simd_load(D_col.add(i) as _);
                let C10 = simd_load(C_col.add(i) as _);
                let D00 = simd_sl_i16!(D10, D_corner, 1);
                D_corner = D10;

                let q = halfsimd_loadu(query.as_ptr(start_i + i) as _);
                let scores = if M::NUC {
                    halfsimd_lookup1_i16(scores1, q)
                } else {
                    halfsimd_lookup2_i16(scores1, scores2, q)
                };

                let mut D11 = simd_adds_i16(D00, scores);
                let C11 = simd_max_i16(simd_adds_i16(C10, gap_extend), simd_adds_i16(D10, gap_open));
                D11 = simd_max_i16(D11, C11);

                let trace_D_C = if TRACE {
                    simd_movemask_i8(simd_cmpeq_i16(D11, C11))
                } else {
                    0 // should be optimized out
                };

                let D11_open = simd_adds_i16(D11, gap_open);
                let mut R11 = simd_sl_i16!(D11_open, R_insert, 1);
                // avoid doing prefix scan if possible!
                if simd_movemask_i8(simd_cmpgt_i16(R11, D11_open)) != 0 {
                    R11 = simd_prefix_scan_i16(R11, P::GAP_EXTEND as i16);
                    D11 = simd_max_i16(D11, R11);
                }
                R_insert = simd_max_i16(D11_open, simd_adds_i16(R11, gap_extend));

                if TRACE {
                    let trace_D_R = simd_movemask_i8(simd_cmpeq_i16(D11, R11));
                    //self.trace.add(((trace_D_R & Self::EVEN_BITS) << 1) | (trace_D_C & Self::EVEN_BITS));
                }

                D_max = simd_max_i16(D_max, D11);
                right_max = simd_max_i16(right_max, D11);

                if X_DROP {
                    let mask = simd_cmpeq_i16(D_max, D11);
                    D_argmax = simd_blend_i8(D_argmax, curr_i, mask);
                    curr_i = simd_adds_i16(curr_i, simd_set1_i16(1));
                }

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

                simd_store(D_col.add(i) as _, D11);
                simd_store(C_col.add(i) as _, C11);
            }

            ptr::write(D_row.add(j), *D_col.add(height - 1));
            // must subtract gap_extend from R_insert due to how R_insert is calculated
            ptr::write(R_row.add(j), simd_extract_i16::<{ L - 1 }>(simd_subs_i16(R_insert, gap_extend)));

            if !X_DROP && start_i + height > query.len()
                && start_j + j >= reference.len() {
                break;
            }
        }

        (D_max, D_argmax, simd_hmax_i16(right_max))
    }

    #[inline(always)]
    pub fn res(&self) -> AlignResult {
        self.res
    }

    #[inline(always)]
    pub fn trace(&self) -> &Trace {
        assert!(TRACE);
        &self.trace
    }

    #[cfg_attr(any(target_arch = "x86", target_arch = "x86_64"), target_feature(enable = "avx2"))]
    #[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
    #[inline]
    unsafe fn get_const_simd(&self) -> (Simd, Simd, Simd) {
        // some useful constant simd vectors
        let neg_inf = simd_set1_i16(i16::MIN);
        let gap_open = simd_set1_i16(P::GAP_OPEN as i16);
        let gap_extend = simd_set1_i16(P::GAP_EXTEND as i16);
        (neg_inf, gap_open, gap_extend)
    }
}

#[inline(always)]
fn convert_char(c: u8, nuc: bool) -> u8 {
    let c = c.to_ascii_uppercase();
    debug_assert!(c >= b'A' && c <= NULL);
    if nuc { c } else { c - b'A' }
}

#[inline(always)]
fn clamp(x: i32) -> i16 {
    cmp::min(cmp::max(x, i16::MIN as i32), i16::MAX as i32) as i16
}

#[inline(always)]
fn div_ceil(n: usize, d: usize) -> usize {
    (n + d - 1) / d
}

pub struct Aligned {
    layout: alloc::Layout,
    ptr: *const i16,
    block_size: usize
}

impl Aligned {
    #[cfg_attr(any(target_arch = "x86", target_arch = "x86_64"), target_feature(enable = "avx2"))]
    #[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
    #[inline]
    pub unsafe fn new(block_size: usize) -> Self {
        let layout = alloc::Layout::from_size_align_unchecked(block_size * 2, L_BYTES);
        let ptr = alloc::alloc(layout) as *const i16;
        let neg_inf = simd_set1_i16(i16::MIN);
        for i in (0..block_size).step_by(L) {
            simd_store(ptr.add(i) as _, neg_inf);
        }
        Self { layout, ptr, block_size }
    }

    #[cfg_attr(any(target_arch = "x86", target_arch = "x86_64"), target_feature(enable = "avx2"))]
    #[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
    #[inline]
    pub unsafe fn set_all(&mut self, o: &Aligned) {
        let o_ptr = o.as_ptr();
        for i in (0..self.block_size).step_by(L) {
            simd_store(self.ptr.add(i) as _, simd_load(o_ptr.add(i) as _));
        }
    }

    #[inline(always)]
    pub fn get(&self, i: usize) -> i16 {
        unsafe { *self.ptr.add(i) }
    }

    #[inline(always)]
    pub fn set(&mut self, i: usize, v: i16) {
        unsafe { ptr::write(self.ptr.add(i) as _, v); }
    }

    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut i16 {
        self.ptr as _
    }

    #[inline(always)]
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
    #[inline(always)]
    pub fn from_bytes(b: &[u8], block_size: usize, nuc: bool) -> Self {
        let mut v = b.to_owned();
        let len = v.len();
        v.insert(0, NULL);
        v.resize(v.len() + block_size, NULL);
        v.iter_mut().for_each(|c| *c = convert_char(*c, nuc));
        Self { s: v, len }
    }

    #[inline(always)]
    pub fn from_str(s: &str, blocks: usize, nuc: bool) -> Self {
        Self::from_bytes(s.as_bytes(), blocks, nuc)
    }

    #[inline(always)]
    pub fn from_string(s: String, block_size: usize, nuc: bool) -> Self {
        let mut v = s.into_bytes();
        let len = v.len();
        v.insert(0, NULL);
        v.resize(v.len() + block_size, NULL);
        v.iter_mut().for_each(|c| *c = convert_char(*c, nuc));
        Self { s: v, len }
    }

    #[inline(always)]
    pub fn get(&self, i: usize) -> u8 {
        unsafe { *self.s.get_unchecked(i) }
    }

    #[inline(always)]
    pub fn set(&mut self, i: usize, c: u8) {
        unsafe { *self.s.get_unchecked_mut(i) = c; }
    }

    #[inline(always)]
    pub fn as_ptr(&self, i: usize) -> *const u8 {
        unsafe { self.s.as_ptr().add(i) }
    }

    #[inline(always)]
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

#[derive(Clone)]
pub struct Trace {
    trace: Vec<u32>,
    shift_dir: Vec<u32>,
    idx: usize
}

impl Trace {
    #[inline(always)]
    pub fn new(query_len: usize, reference_len: usize) -> Self {
        let len = query_len + reference_len;
        Self {
            trace: vec![0; div_ceil(len, 16)],
            shift_dir: vec![0; div_ceil(div_ceil(len, L), 16)],
            idx: 0
        }
    }

    #[inline(always)]
    pub fn add(&mut self, t: u32) {
        unsafe { *self.trace.get_unchecked_mut(self.idx) = t; }
        self.idx += 1;
    }

    #[inline(always)]
    pub fn dir(&mut self, d: u32) {
        let i = self.idx / L;
        unsafe {
            *self.shift_dir.get_unchecked_mut(i / 16) |= d << (i % 16);
        }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.trace.fill(0);
        self.shift_dir.fill(0);
        self.idx = 0;
    }
}

#[cfg(test)]
mod tests {
    use crate::scores::*;

    use super::*;

    #[test]
    fn test_no_x_drop() {
        type TestParams = GapParams<-11, -1>;

        let r = PaddedBytes::from_bytes(b"AAAA", 16, false);
        let q = PaddedBytes::from_bytes(b"AARA", 16, false);
        let a = Block::<TestParams, _, 16, 16, false, false>::align(&q, &r, &BLOSUM62, 0, 0, 0);
        assert_eq!(a.res().score, 11);

        let r = PaddedBytes::from_bytes(b"AAAA", 16, false);
        let q = PaddedBytes::from_bytes(b"AAAA", 16, false);
        let a = Block::<TestParams, _, 16, 16, false, false>::align(&q, &r, &BLOSUM62, 0, 0, 0);
        assert_eq!(a.res().score, 16);

        let r = PaddedBytes::from_bytes(b"AAAA", 16, false);
        let q = PaddedBytes::from_bytes(b"AARA", 16, false);
        let a = Block::<TestParams, _, 16, 16, false, false>::align(&q, &r, &BLOSUM62, 0, 0, 0);
        assert_eq!(a.res().score, 11);

        let r = PaddedBytes::from_bytes(b"AAAA", 16, false);
        let q = PaddedBytes::from_bytes(b"RRRR", 16, false);
        let a = Block::<TestParams, _, 16, 16, false, false>::align(&q, &r, &BLOSUM62, 0, 0, 0);
        assert_eq!(a.res().score, -4);

        let r = PaddedBytes::from_bytes(b"AAAA", 16, false);
        let q = PaddedBytes::from_bytes(b"AAA", 16, false);
        let a = Block::<TestParams, _, 16, 16, false, false>::align(&q, &r, &BLOSUM62, 0, 0, 0);
        assert_eq!(a.res().score, 1);

        type TestParams2 = GapParams<-1, -1>;

        let r = PaddedBytes::from_bytes(b"AAAN", 16, true);
        let q = PaddedBytes::from_bytes(b"ATAA", 16, true);
        let a = Block::<TestParams2, _, 16, 16, false, false>::align(&q, &r, &NW1, 0, 0, 0);
        assert_eq!(a.res().score, 1);

        let r = PaddedBytes::from_bytes(b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", 16, true);
        let q = PaddedBytes::from_bytes(b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", 16, true);
        let a = Block::<TestParams2, _, 16, 16, false, false>::align(&q, &r, &NW1, 0, 0, 0);
        assert_eq!(a.res().score, 32);

        let r = PaddedBytes::from_bytes(b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", 16, true);
        let q = PaddedBytes::from_bytes(b"TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT", 16, true);
        let a = Block::<TestParams2, _, 16, 16, false, false>::align(&q, &r, &NW1, 0, 0, 0);
        assert_eq!(a.res().score, -32);

        let r = PaddedBytes::from_bytes(b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", 16, true);
        let q = PaddedBytes::from_bytes(b"TATATATATATATATATATATATATATATATA", 16, true);
        let a = Block::<TestParams2, _, 16, 16, false, false>::align(&q, &r, &NW1, 0, 0, 0);
        assert_eq!(a.res().score, 0);

        let r = PaddedBytes::from_bytes(b"TTAAAAAAATTTTTTTTTTTT", 16, true);
        let q = PaddedBytes::from_bytes(b"TTTTTTTTAAAAAAATTTTTTTTT", 16, true);
        let a = Block::<TestParams2, _, 16, 16, false, false>::align(&q, &r, &NW1, 0, 0, 0);
        assert_eq!(a.res().score, 9);

        let r = PaddedBytes::from_bytes(b"AAAA", 16, true);
        let q = PaddedBytes::from_bytes(b"C", 16, true);
        let a = Block::<TestParams2, _, 16, 16, false, false>::align(&q, &r, &NW1, 0, 0, 0);
        assert_eq!(a.res().score, -4);
        let a = Block::<TestParams2, _, 16, 16, false, false>::align(&r, &q, &NW1, 0, 0, 0);
        assert_eq!(a.res().score, -4);
    }

    #[test]
    fn test_x_drop() {
        type TestParams = GapParams<-11, -1>;

        let r = PaddedBytes::from_bytes(b"AAARRA", 16, false);
        let q = PaddedBytes::from_bytes(b"AAAAAA", 16, false);
        let a = Block::<TestParams, _, 16, 16, false, true>::align(&q, &r, &BLOSUM62, 1, 0, 0);
        assert_eq!(a.res(), AlignResult { score: 14, query_idx: 6, reference_idx: 6 });

        let r = PaddedBytes::from_bytes(b"AAAAAAAAAAAAAAARRRRRRRRRRRRRRRRAAAAAAAAAAAAA", 16, false);
        let q = PaddedBytes::from_bytes(b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", 16, false);
        let a = Block::<TestParams, _, 16, 16, false, true>::align(&q, &r, &BLOSUM62, 1, 0, 0);
        assert_eq!(a.res(), AlignResult { score: 60, query_idx: 15, reference_idx: 15 });
    }
}
