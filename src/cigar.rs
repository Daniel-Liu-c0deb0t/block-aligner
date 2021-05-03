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
    s: Vec<OpLen>
}

impl Cigar {
    pub unsafe fn new(max_len: usize) -> Self {
        let s = Vec::with_capacity(max_len);
        Cigar { s }
    }

    pub unsafe fn add(&mut self, op: Operation) {
        debug_assert!(self.s.len() < self.s.capacity());
        if self.s.len() == 0 || op != self.s.get_unchecked(self.s.len() - 1).op {
            let idx = self.s.len();
            self.s.set_len(self.s.len() + 1);
            *self.s.get_unchecked_mut(idx) = OpLen { op, len: 1 };
        } else {
            let idx = self.s.len() - 1;
            self.s.get_unchecked_mut(idx).len += 1;
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
            write!(f, "{}{}", op_len.len, c)?;
        }
        Ok(())
    }
}
