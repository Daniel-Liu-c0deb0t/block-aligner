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
        let len = self.s.len();
        debug_assert!(len < self.s.capacity());
        // almost branchless
        let add = if len == 0 { 1 } else { (op != self.s.get_unchecked(len - 1).op) as usize };
        let idx = len + add;
        self.s.set_len(idx);
        *self.s.get_unchecked_mut(len) = OpLen { op, len: 0 };
        self.s.get_unchecked_mut(idx - 1).len += 1;
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
