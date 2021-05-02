use std::fmt;

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Operation {
    M,
    I,
    D
}

#[derive(Debug, Copy, Clone)]
struct OpLen {
    op: Operation,
    len: usize
}

pub struct Cigar {
    s: Vec<OpLen>,
    idx: usize
}

impl Cigar {
    pub unsafe fn new(max_len: usize) -> Self {
        let mut s = Vec::with_capacity(max_len);
        s.set_len(max_len);
        Cigar { s, idx: 0 }
    }

    pub unsafe fn add(&mut self, op: Operation) {
        if idx == 0 || op != self.s.get_unchecked(self.idx - 1).0 {
            *self.s.get_unchecked_mut(self.idx) = OpLen { op, len: 1 };
            self.idx += 1;
        } else {
            *self.s.get_unchecked_mut(self.idx - 1).1 += 1;
        }
    }
}

impl fmt::Display for Cigar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for &op_len in self.s.iter().rev() {
            let c = match op_len.op {
                Operation::M => 'M',
                Operation::I => 'I',
                Operation::D => 'D'
            };
            write!(f, "{}{}", op_len.len, c);
        }
        Ok(())
    }
}
