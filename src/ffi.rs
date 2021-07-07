//! C bindings for block aligner.
//!
//! Generics are monomorphised manually.
//!
//! Nucleotide and arbitrary byte alignment do not have bindings yet.

use std::ffi::{CStr, c_void};
use std::os::raw::c_char;

use crate::scan_block::*;
use crate::scores::*;
use crate::cigar::OpLen;

// avoid generics by using void pointer and monomorphism
/// A handle for a block in block aligner.
pub type BlockHandle = *mut c_void;

/// Represents a range that has inclusive lower and upper bounds.
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub struct SizeRange {
    pub min: usize,
    pub max: usize
}

/// Represents a vector of operations that form a CIGAR string.
///
/// The length and capacity fields should not be modified, since
/// they are needed to correctly free the CIGAR string.
#[repr(C)]
pub struct CigarVec {
    pub ptr: *mut OpLen,
    pub len: usize,
    pub cap: usize
}

/// Create a padded amino acid string.
#[no_mangle]
pub unsafe extern fn block_make_padded_aa(s: *const c_char, max_size: usize) -> *mut PaddedBytes {
    let c_str = CStr::from_ptr(s);
    let padded_bytes = Box::new(PaddedBytes::from_bytes::<AAMatrix>(c_str.to_bytes(), max_size));
    Box::into_raw(padded_bytes)
}

/// Frees a padded amino acid string.
#[no_mangle]
pub unsafe extern fn block_free_padded_aa(padded: *mut PaddedBytes) {
    drop(Box::from_raw(padded));
}

/// Frees a cigar vector.
#[no_mangle]
pub unsafe extern fn block_free_cigar(v: CigarVec) {
    drop(Vec::from_raw_parts(v.ptr, v.len, v.cap));
}

// No traceback

/// Global alignment of two amino acid strings (no traceback).
#[no_mangle]
pub unsafe extern fn block_align_aa(q: *const PaddedBytes,
                                    r: *const PaddedBytes,
                                    m: *const AAMatrix,
                                    g: Gaps,
                                    s: SizeRange) -> BlockHandle {
    let aligner = Box::new(Block::<_, false, false>::align(&*q, &*r, &*m, g, s.min..=s.max, 0));
    Box::into_raw(aligner) as BlockHandle
}

/// Retrieves the result of global alignment of two amino acid strings (no traceback).
#[no_mangle]
pub unsafe extern fn block_res_aa(b: BlockHandle) -> AlignResult {
    let aligner = &*(b as *const Block<AAMatrix, false, false>);
    aligner.res()
}

/// Frees the block used for global alignment of two amino acid strings (no traceback).
#[no_mangle]
pub unsafe extern fn block_free_aa(b: BlockHandle) {
    drop(Box::from_raw(b as *mut Block<AAMatrix, false, false>));
}

/// X-drop alignment of two amino acid strings (no traceback).
#[no_mangle]
pub unsafe extern fn block_align_aa_xdrop(q: *const PaddedBytes,
                                          r: *const PaddedBytes,
                                          m: *const AAMatrix,
                                          g: Gaps,
                                          s: SizeRange,
                                          x: i32) -> BlockHandle {
    let aligner = Box::new(Block::<_, false, true>::align(&*q, &*r, &*m, g, s.min..=s.max, x));
    Box::into_raw(aligner) as BlockHandle
}

/// Retrieves the result of X-drop alignment of two amino acid strings (no traceback).
#[no_mangle]
pub unsafe extern fn block_res_aa_xdrop(b: BlockHandle) -> AlignResult {
    let aligner = &*(b as *const Block<AAMatrix, false, true>);
    aligner.res()
}

/// Frees the block used for X-drop alignment of two amino acid strings (no traceback).
#[no_mangle]
pub unsafe extern fn block_free_aa_xdrop(b: BlockHandle) {
    drop(Box::from_raw(b as *mut Block<AAMatrix, false, true>));
}

// With traceback

/// Global alignment of two amino acid strings, with traceback.
#[no_mangle]
pub unsafe extern fn block_align_aa_trace(q: *const PaddedBytes,
                                          r: *const PaddedBytes,
                                          m: *const AAMatrix,
                                          g: Gaps,
                                          s: SizeRange) -> BlockHandle {
    let aligner = Box::new(Block::<_, true, false>::align(&*q, &*r, &*m, g, s.min..=s.max, 0));
    Box::into_raw(aligner) as BlockHandle
}

/// Retrieves the result of global alignment of two amino acid strings, with traceback.
#[no_mangle]
pub unsafe extern fn block_res_aa_trace(b: BlockHandle) -> AlignResult {
    let aligner = &*(b as *const Block<AAMatrix, true, false>);
    aligner.res()
}

/// Retrieves the resulting CIGAR string from global alignment of two amino acid strings, with traceback.
#[no_mangle]
pub unsafe extern fn block_cigar_aa_trace(b: BlockHandle) -> CigarVec {
    let aligner = &*(b as *const Block<AAMatrix, true, false>);
    let res = aligner.res();
    let cigar_vec = aligner.trace().cigar(res.query_idx, res.reference_idx).to_vec();
    let (ptr, len, cap) = cigar_vec.into_raw_parts();
    CigarVec { ptr, len, cap }
}

/// Frees the block used for global alignment of two amino acid strings, with traceback.
#[no_mangle]
pub unsafe extern fn block_free_aa_trace(b: BlockHandle) {
    drop(Box::from_raw(b as *mut Block<AAMatrix, true, false>));
}

/// X-drop alignment of two amino acid strings, with traceback.
#[no_mangle]
pub unsafe extern fn block_align_aa_trace_xdrop(q: *const PaddedBytes,
                                                r: *const PaddedBytes,
                                                m: *const AAMatrix,
                                                g: Gaps,
                                                s: SizeRange,
                                                x: i32) -> BlockHandle {
    let aligner = Box::new(Block::<_, true, true>::align(&*q, &*r, &*m, g, s.min..=s.max, x));
    Box::into_raw(aligner) as BlockHandle
}

/// Retrieves the result of X-drop alignment of two amino acid strings, with traceback.
#[no_mangle]
pub unsafe extern fn block_res_aa_trace_xdrop(b: BlockHandle) -> AlignResult {
    let aligner = &*(b as *const Block<AAMatrix, true, true>);
    aligner.res()
}

/// Retrieves the resulting CIGAR string from X-drop alignment of two amino acid strings, with traceback.
#[no_mangle]
pub unsafe extern fn block_cigar_aa_trace_xdrop(b: BlockHandle) -> CigarVec {
    let aligner = &*(b as *const Block<AAMatrix, true, true>);
    let res = aligner.res();
    let cigar_vec = aligner.trace().cigar(res.query_idx, res.reference_idx).to_vec();
    let (ptr, len, cap) = cigar_vec.into_raw_parts();
    CigarVec { ptr, len, cap }
}

/// Frees the block used for X-drop alignment of two amino acid strings, with traceback.
#[no_mangle]
pub unsafe extern fn block_free_aa_trace_xdrop(b: BlockHandle) {
    drop(Box::from_raw(b as *mut Block<AAMatrix, true, true>));
}
