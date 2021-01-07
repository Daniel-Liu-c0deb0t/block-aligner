#[repr(align(32))]
pub struct Scores {
    pub data: [i8; 27 * 32]
}

impl Scores {
    #[inline]
    unsafe fn as_ptr(&self, i: usize) -> *const i8 {
        debug_assert!(i < 27);
        self.data.as_ptr().offset((i as isize) * 32)
    }
}

pub static BLOSUM62: Scores = Scores { data: include!("../matrices/BLOSUM62") };

pub struct Matrix {
    scores: Scores,
    gap_open: i8,
    gap_extend: i8
}

impl Matrix {
    pub fn new(scores: Scores, gap_open: i8, gap_extend: i8) -> Self {
        Matrix { scores, gap_open, gap_extend }
    }

    #[inline]
    pub fn as_ptr(&self, i: usize) -> *const i8 {
        unsafe { self.scores.as_ptr(i) }
    }

    #[inline]
    pub fn gap_open(&self) -> i8 {
        self.gap_open
    }

    #[inline]
    pub fn gap_extend(&self) -> i8 {
        self.gap_extend
    }
}
