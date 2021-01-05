#[repr(align(32))]
pub struct Scores {
    pub data: [u8; 27 * 32]
}

impl Scores {
    #[inline]
    fn get(&self, i: usize) -> *const u8 {
        debug_assert!(i >= 0 && i < 27);
        self.data.as_ptr().offset((i as isize) * 32)
    }
}

static BLOSUM62 = Scores { data: [include!("../matrices/BLOSUM62")] };

pub struct Matrix {
    scores: Scores,
    gap_open: i8,
    gap_extend: i8
}

impl Matrix {
    pub fn new(scores: Scores, gap_open: i8, gap_extend: i8) {
        Matrix { scores, gap_open, gap_extend }
    }

    #[inline]
    pub fn get(&self, i: usize) -> *const u8 {
        self.scores.get(i)
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
