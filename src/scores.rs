//! Structs for representing match/mismatch scoring matrices.

#[cfg(feature = "simd_avx2")]
use crate::avx2::*;

#[cfg(feature = "simd_wasm")]
use crate::simd128::*;

use std::i8;

pub trait Matrix {
    /// Byte to use as padding.
    const NULL: u8;
    /// Create a new matrix with default (usually nonsense) values.
    ///
    /// Use `new_simple` to create a sensible scoring matrix.
    fn new() -> Self;
    /// Set the score for a pair of bytes.
    fn set(&mut self, a: u8, b: u8, score: i8);
    /// Get the score for a pair of bytes.
    fn get(&self, a: u8, b: u8) -> i8;
    /// Get the pointer for a specific index.
    fn as_ptr(&self, i: usize) -> *const i8;
    /// Get the scores for a certain byte and a certain SIMD vector of bytes.
    unsafe fn get_scores(&self, c: u8, v: HalfSimd, right: bool) -> Simd;
    /// Convert a byte to a better storage format that makes retrieving scores
    /// easier.
    fn convert_char(c: u8) -> u8;
}

/// Amino acid scoring matrix.
#[repr(C, align(32))]
#[derive(Clone, PartialEq, Debug)]
pub struct AAMatrix {
    scores: [i8; 27 * 32]
}

impl AAMatrix {
    /// Create a simple matrix with a certain match and mismatch score.
    pub const fn new_simple(match_score: i8, mismatch_score: i8) -> Self {
        let mut scores = [i8::MIN; 27 * 32];
        let mut i = b'A';
        while i <= b'Z' {
            let mut j = b'A';
            while j <= b'Z' {
                let idx = ((i - b'A') as usize) * 32 + ((j - b'A') as usize);
                scores[idx] = if i == j { match_score } else { mismatch_score };
                j += 1;
            }
            i += 1;
        }
        Self { scores }
    }
}

impl Matrix for AAMatrix {
    const NULL: u8 = b'A' + 26u8;

    fn new() -> Self {
        Self { scores: [i8::MIN; 27 * 32] }
    }

    fn set(&mut self, a: u8, b: u8, score: i8) {
        let a = a.to_ascii_uppercase();
        let b = b.to_ascii_uppercase();
        assert!(b'A' <= a && a <= b'Z' + 1);
        assert!(b'A' <= b && b <= b'Z' + 1);
        let idx = ((a - b'A') as usize) * 32 + ((b - b'A') as usize);
        self.scores[idx] = score;
        let idx = ((b - b'A') as usize) * 32 + ((a - b'A') as usize);
        self.scores[idx] = score;
    }

    fn get(&self, a: u8, b: u8) -> i8 {
        let a = a.to_ascii_uppercase();
        let b = b.to_ascii_uppercase();
        assert!(b'A' <= a && a <= b'Z' + 1);
        assert!(b'A' <= b && b <= b'Z' + 1);
        let idx = ((a - b'A') as usize) * 32 + ((b - b'A') as usize);
        self.scores[idx]
    }

    #[inline]
    fn as_ptr(&self, i: usize) -> *const i8 {
        debug_assert!(i < 27);
        unsafe { self.scores.as_ptr().add(i * 32) }
    }

    #[cfg_attr(feature = "simd_avx2", target_feature(enable = "avx2"))]
    #[cfg_attr(feature = "simd_wasm", target_feature(enable = "simd128"))]
    #[inline]
    unsafe fn get_scores(&self, c: u8, v: HalfSimd, _right: bool) -> Simd {
        // efficiently lookup scores for each character in v
        let matrix_ptr = self.as_ptr(c as usize);
        let scores1 = halfsimd_load(matrix_ptr as *const HalfSimd);
        let scores2 = halfsimd_load((matrix_ptr as *const HalfSimd).add(1));
        halfsimd_lookup2_i16(scores1, scores2, v)
    }

    #[inline]
    fn convert_char(c: u8) -> u8 {
        let c = c.to_ascii_uppercase();
        assert!(c >= b'A' && c <= Self::NULL);
        c - b'A'
    }
}

/// Nucleotide scoring matrix.
#[repr(C, align(32))]
#[derive(Clone, PartialEq, Debug)]
pub struct NucMatrix {
    scores: [i8; 8 * 16]
}

impl NucMatrix {
    /// Create a simple matrix with a certain match and mismatch score.
    pub const fn new_simple(match_score: i8, mismatch_score: i8) -> Self {
        let mut scores = [i8::MIN; 8 * 16];
        let alpha = [b'A', b'T', b'C', b'G', b'N'];
        let mut i = 0;
        while i < alpha.len() {
            let mut j = 0;
            while j < alpha.len() {
                let idx = ((alpha[i] & 0b111) as usize) * 16 + ((alpha[j] & 0b1111) as usize);
                scores[idx] = if i == j { match_score } else { mismatch_score };
                j += 1;
            }
            i += 1;
        }
        Self { scores }
    }
}

impl Matrix for NucMatrix {
    const NULL: u8 = b'Z';

    fn new() -> Self {
        Self { scores: [i8::MIN; 8 * 16] }
    }

    fn set(&mut self, a: u8, b: u8, score: i8) {
        let a = a.to_ascii_uppercase();
        let b = b.to_ascii_uppercase();
        assert!(b'A' <= a && a <= b'Z');
        assert!(b'A' <= b && b <= b'Z');
        let idx = ((a & 0b111) as usize) * 16 + ((b & 0b1111) as usize);
        self.scores[idx] = score;
        let idx = ((b & 0b111) as usize) * 16 + ((a & 0b1111) as usize);
        self.scores[idx] = score;
    }

    fn get(&self, a: u8, b: u8) -> i8 {
        let a = a.to_ascii_uppercase();
        let b = b.to_ascii_uppercase();
        assert!(b'A' <= a && a <= b'Z');
        assert!(b'A' <= b && b <= b'Z');
        let idx = ((a & 0b111) as usize) * 16 + ((b & 0b1111) as usize);
        self.scores[idx]
    }

    #[inline]
    fn as_ptr(&self, i: usize) -> *const i8 {
        unsafe { self.scores.as_ptr().add((i & 0b111) * 16) }
    }

    #[cfg_attr(feature = "simd_avx2", target_feature(enable = "avx2"))]
    #[cfg_attr(feature = "simd_wasm", target_feature(enable = "simd128"))]
    #[inline]
    unsafe fn get_scores(&self, c: u8, v: HalfSimd, _right: bool) -> Simd {
        // efficiently lookup scores for each character in v
        let matrix_ptr = self.as_ptr(c as usize);
        let scores = halfsimd_load(matrix_ptr as *const HalfSimd);
        halfsimd_lookup1_i16(scores, v)
    }

    #[inline]
    fn convert_char(c: u8) -> u8 {
        let c = c.to_ascii_uppercase();
        assert!(c >= b'A' && c <= Self::NULL);
        c
    }
}

/// Arbitrary bytes scoring matrix.
#[repr(C)]
#[derive(Clone, PartialEq, Debug)]
pub struct ByteMatrix {
    match_score: i8,
    mismatch_score: i8
}

impl ByteMatrix {
    /// Create a simple matrix with a certain match and mismatch score.
    pub const fn new_simple(match_score: i8, mismatch_score: i8) -> Self {
        Self { match_score, mismatch_score }
    }
}

impl Matrix for ByteMatrix {
    /// May lead to inaccurate results with x drop alignment,
    /// if the block reaches the ends of the strings.
    ///
    /// Avoid using `ByteMatrix` with x drop alignment.
    const NULL: u8 = b'\0';

    fn new() -> Self {
        Self { match_score: i8::MIN, mismatch_score: i8::MIN }
    }

    fn set(&mut self, _a: u8, _b: u8, _score: i8) {
        unimplemented!();
    }

    fn get(&self, a: u8, b: u8) -> i8 {
        if a == b { self.match_score } else { self.mismatch_score }
    }

    #[inline]
    fn as_ptr(&self, _i: usize) -> *const i8 {
        unimplemented!()
    }

    #[cfg_attr(feature = "simd_avx2", target_feature(enable = "avx2"))]
    #[cfg_attr(feature = "simd_wasm", target_feature(enable = "simd128"))]
    #[inline]
    unsafe fn get_scores(&self, c: u8, v: HalfSimd, _right: bool) -> Simd {
        let match_scores = halfsimd_set1_i8(self.match_score);
        let mismatch_scores = halfsimd_set1_i8(self.mismatch_score);
        halfsimd_lookup_bytes_i16(match_scores, mismatch_scores, halfsimd_set1_i8(c as i8), v)
    }

    #[inline]
    fn convert_char(c: u8) -> u8 {
        c
    }
}

/// Match = 1, mismatch = -1.
#[cfg_attr(not(target_arch = "wasm32"), no_mangle)]
pub static NW1: NucMatrix = NucMatrix::new_simple(1, -1);

#[cfg_attr(not(target_arch = "wasm32"), no_mangle)]
pub static BLOSUM45: AAMatrix = AAMatrix { scores: include!("../matrices/BLOSUM45") };

#[cfg_attr(not(target_arch = "wasm32"), no_mangle)]
pub static BLOSUM50: AAMatrix = AAMatrix { scores: include!("../matrices/BLOSUM50") };

#[cfg_attr(not(target_arch = "wasm32"), no_mangle)]
pub static BLOSUM62: AAMatrix = AAMatrix { scores: include!("../matrices/BLOSUM62") };

#[cfg_attr(not(target_arch = "wasm32"), no_mangle)]
pub static BLOSUM80: AAMatrix = AAMatrix { scores: include!("../matrices/BLOSUM80") };

#[cfg_attr(not(target_arch = "wasm32"), no_mangle)]
pub static BLOSUM90: AAMatrix = AAMatrix { scores: include!("../matrices/BLOSUM90") };

#[cfg_attr(not(target_arch = "wasm32"), no_mangle)]
pub static PAM100: AAMatrix = AAMatrix { scores: include!("../matrices/PAM100") };

#[cfg_attr(not(target_arch = "wasm32"), no_mangle)]
pub static PAM120: AAMatrix = AAMatrix { scores: include!("../matrices/PAM120") };

#[cfg_attr(not(target_arch = "wasm32"), no_mangle)]
pub static PAM160: AAMatrix = AAMatrix { scores: include!("../matrices/PAM160") };

#[cfg_attr(not(target_arch = "wasm32"), no_mangle)]
pub static PAM200: AAMatrix = AAMatrix { scores: include!("../matrices/PAM200") };

#[cfg_attr(not(target_arch = "wasm32"), no_mangle)]
pub static PAM250: AAMatrix = AAMatrix { scores: include!("../matrices/PAM250") };

/// Match = 1, mismatch = -1.
#[cfg_attr(not(target_arch = "wasm32"), no_mangle)]
pub static BYTES1: ByteMatrix = ByteMatrix::new_simple(1, -1);

/*pub trait ScoreParams {
    const GAP_OPEN: i8;
    const GAP_EXTEND: i8;
    const I: usize;
}

pub struct Params<const GAP_OPEN: i8, const GAP_EXTEND: i8, const I: usize>;

impl<const GAP_OPEN: i8, const GAP_EXTEND: i8, const I: usize> ScoreParams for Params<{ GAP_OPEN }, { GAP_EXTEND }, { I }> {
    const GAP_OPEN: i8 = GAP_OPEN;
    const GAP_EXTEND: i8 = GAP_EXTEND;
    const I: usize = I;
}

pub type GapParams<const GAP_OPEN: i8, const GAP_EXTEND: i8> = Params<{ GAP_OPEN }, { GAP_EXTEND }, 0>;*/

/// Open and extend gap costs.
///
/// Open cost must include the extend cost. For example, with `Gaps { open: -11, extend: -1 }`,
/// a gap of length 1 costs -11, and a gap of length 2 costs -12.
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct Gaps {
    pub open: i8,
    pub extend: i8
}
