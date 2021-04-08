pub trait Matrix {
    const NUC: bool;

    fn as_ptr(&self, i: usize) -> *const i8;
}

#[repr(align(32))]
#[derive(Clone, PartialEq, Debug)]
pub struct AAMatrix {
    pub scores: [i8; 27 * 32]
}

impl Matrix for AAMatrix {
    const NUC: bool = false;

    #[inline]
    fn as_ptr(&self, i: usize) -> *const i8 {
        debug_assert!(i < 27);
        unsafe { self.scores.as_ptr().add(i * 32) }
    }
}

#[repr(align(32))]
#[derive(Clone, PartialEq, Debug)]
pub struct NucMatrix {
    pub scores: [i8; 8 * 16]
}

impl Matrix for NucMatrix {
    const NUC: bool = true;

    #[inline]
    fn as_ptr(&self, i: usize) -> *const i8 {
        unsafe { self.scores.as_ptr().add((i & 0b111) * 16) }
    }
}

pub static NW1: NucMatrix = NucMatrix { scores: include!("../matrices/NW1") };

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
