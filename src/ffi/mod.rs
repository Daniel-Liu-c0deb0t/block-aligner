#![cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"))]

use std::ffi::{CStr, c_void};
use std::os::raw::c_char;

use crate::scan_block::*;
use crate::scores::*;

pub type BlockHandle = *const c_void;
pub type AAPaddedStrHandle = *const c_void;
pub type AAMatrixHandle = *const c_void;

#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub struct SizeRange {
    min: usize,
    max: usize
}

pub unsafe extern fn block_align_aa_make_padded(s: *const c_char, max_size: usize) -> AAPaddedStrHandle {
    let c_str = CStr::from_ptr(s);
    let padded_bytes = Box::new(PaddedBytes::from_bytes::<AAMatrix>(c_str.to_bytes(), max_size));
    Box::into_raw(padded_bytes) as AAPaddedStrHandle
}

pub unsafe extern fn block_align_aa(q: AAPaddedStrHandle,
                                    r: AAPaddedStrHandle,
                                    m: AAMatrixHandle,
                                    g: Gaps,
                                    s: SizeRange) -> BlockHandle {
    let q = &*(q as *const PaddedBytes);
    let r = &*(r as *const PaddedBytes);
    let m = &*(m as *const AAMatrix);
    let aligner = Box::new(Block::<_, false, false>::align(q, r, m, g, s.min..=s.max, 0));
    Box::into_raw(aligner) as BlockHandle
}

pub unsafe extern fn block_align_aa_xdrop(q: AAPaddedStrHandle,
                                          r: AAPaddedStrHandle,
                                          m: AAMatrixHandle,
                                          g: Gaps,
                                          s: SizeRange,
                                          x: i32) -> BlockHandle {
    let q = &*(q as *const PaddedBytes);
    let r = &*(r as *const PaddedBytes);
    let m = &*(m as *const AAMatrix);
    let aligner = Box::new(Block::<_, false, true>::align(q, r, m, g, s.min..=s.max, x));
    Box::into_raw(aligner) as BlockHandle
}

pub unsafe extern fn block_align_aa_free(b: BlockHandle) {
    drop(Box::from_raw(b as *mut Block<AAMatrix, false, false>));
}

pub unsafe extern fn block_align_aa_xdrop_free(b: BlockHandle) {
    drop(Box::from_raw(b as *mut Block<AAMatrix, false, true>));
}
