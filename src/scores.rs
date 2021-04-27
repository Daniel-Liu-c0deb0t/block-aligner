use std::i8;

pub trait Matrix {
    const NUC: bool;

    fn new() -> Self;
    fn set(&mut self, a: u8, b: u8, score: i8);
    fn get(&self, a: u8, b: u8) -> i8;
    fn as_ptr(&self, i: usize) -> *const i8;
}

#[repr(align(32))]
#[derive(Clone, PartialEq, Debug)]
pub struct AAMatrix {
    scores: [i8; 27 * 32]
}

impl AAMatrix {
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
    const NUC: bool = false;

    fn new() -> Self {
        Self { scores: [i8::MIN; 27 * 32] }
    }

    fn set(&mut self, a: u8, b: u8, score: i8) {
        let a = a.to_ascii_uppercase();
        let b = b.to_ascii_uppercase();
        debug_assert!(b'A' <= a && a <= b'Z' + 1);
        debug_assert!(b'A' <= b && b <= b'Z' + 1);
        let idx = ((a - b'A') as usize) * 32 + ((b - b'A') as usize);
        self.scores[idx] = score;
        let idx = ((b - b'A') as usize) * 32 + ((a - b'A') as usize);
        self.scores[idx] = score;
    }

    fn get(&self, a: u8, b: u8) -> i8 {
        let a = a.to_ascii_uppercase();
        let b = b.to_ascii_uppercase();
        debug_assert!(b'A' <= a && a <= b'Z' + 1);
        debug_assert!(b'A' <= b && b <= b'Z' + 1);
        let idx = ((a - b'A') as usize) * 32 + ((b - b'A') as usize);
        self.scores[idx]
    }

    #[inline(always)]
    fn as_ptr(&self, i: usize) -> *const i8 {
        debug_assert!(i < 27);
        unsafe { self.scores.as_ptr().add(i * 32) }
    }
}

#[repr(align(32))]
#[derive(Clone, PartialEq, Debug)]
pub struct NucMatrix {
    scores: [i8; 8 * 16]
}

impl NucMatrix {
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
    const NUC: bool = true;

    fn new() -> Self {
        Self { scores: [i8::MIN; 8 * 16] }
    }

    fn set(&mut self, a: u8, b: u8, score: i8) {
        let a = a.to_ascii_uppercase();
        let b = b.to_ascii_uppercase();
        debug_assert!(b'A' <= a && a <= b'Z');
        debug_assert!(b'A' <= b && b <= b'Z');
        let idx = ((a & 0b111) as usize) * 16 + ((b & 0b1111) as usize);
        self.scores[idx] = score;
        let idx = ((b & 0b111) as usize) * 16 + ((a & 0b1111) as usize);
        self.scores[idx] = score;
    }

    fn get(&self, a: u8, b: u8) -> i8 {
        let a = a.to_ascii_uppercase();
        let b = b.to_ascii_uppercase();
        debug_assert!(b'A' <= a && a <= b'Z');
        debug_assert!(b'A' <= b && b <= b'Z');
        let idx = ((a & 0b111) as usize) * 16 + ((b & 0b1111) as usize);
        self.scores[idx]
    }

    #[inline(always)]
    fn as_ptr(&self, i: usize) -> *const i8 {
        unsafe { self.scores.as_ptr().add((i & 0b111) * 16) }
    }
}

pub static NW1: NucMatrix = NucMatrix::new_simple(1, -1);

pub static BLOSUM62: AAMatrix = AAMatrix { scores: include!("../matrices/BLOSUM62") };

pub trait ScoreParams {
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

pub type GapParams<const GAP_OPEN: i8, const GAP_EXTEND: i8> = Params<{ GAP_OPEN }, { GAP_EXTEND }, 0>;
