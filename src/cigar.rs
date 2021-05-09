use std::fmt;

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Operation {
    Sentinel,
    M,
    I,
    D
}

#[derive(Debug, Copy, Clone)]
pub struct OpLen {
    pub op: Operation,
    pub len: usize
}

pub struct Cigar {
    s: Vec<OpLen>,
    idx: usize
}

impl Cigar {
    pub unsafe fn new(max_len: usize) -> Self {
        let s = vec![OpLen { op: Operation::Sentinel, len: 0 }; max_len + 1];
        let idx = 1;
        Cigar { s, idx }
    }

    pub unsafe fn add(&mut self, op: Operation) {
        debug_assert!(self.idx < self.s.len());
        // branchlessly append one operation
        let add = (op != self.s.get_unchecked(self.idx - 1).op) as usize;
        self.idx += add;
        self.s.get_unchecked_mut(self.idx - 1).op = op;
        self.s.get_unchecked_mut(self.idx - 1).len += 1;
    }

    pub fn len(&self) -> usize {
        self.idx - 1
    }

    pub fn get(&self, i: usize) -> OpLen {
        self.s[self.idx - 1 - i]
    }
}

impl fmt::Display for Cigar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for &op_len in self.s.iter().rev() {
            let c = match op_len.op {
                Operation::M => 'M',
                Operation::I => 'I',
                Operation::D => 'D',
                _ => continue
            };
            write!(f, "{}{}", op_len.len, c)?;
        }
        Ok(())
    }
}
