#[repr(align(32))]
#[derive(Clone)]
pub struct SubMatrix {
    pub scores: [i8; 27 * 32]
}

impl SubMatrix {
    #[inline]
    pub fn as_ptr(&self, i: usize) -> *const i8 {
        debug_assert!(i < 27);
        unsafe { self.scores.as_ptr().add(i * 32) }
    }
}

pub static BLOSUM62: SubMatrix = SubMatrix { scores: include!("../matrices/BLOSUM62") };

pub trait GapScores {
    const GAP_OPEN: i8;
    const GAP_EXTEND: i8;
}

pub struct Gap<const GAP_OPEN: i8, const GAP_EXTEND: i8>;

impl<const GAP_OPEN: i8, const GAP_EXTEND: i8> GapScores for Gap<{ GAP_OPEN }, { GAP_EXTEND }> {
    const GAP_OPEN: i8 = GAP_OPEN;
    const GAP_EXTEND: i8 = GAP_EXTEND;
}
