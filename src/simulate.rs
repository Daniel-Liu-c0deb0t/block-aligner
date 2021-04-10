use rand::prelude::*;

pub static AMINO_ACIDS: [u8; 20] = [
    b'A', b'C', b'D', b'E', b'F', b'G', b'H', b'I', b'K', b'L',
    b'M', b'N', b'P', b'Q', b'R', b'S', b'T', b'V', b'W', b'Y'
];

pub static NUC: [u8; 5] = [
    b'A', b'C', b'G', b'N', b'T'
];

pub fn rand_mutate<R: Rng>(a: &[u8], k: usize, alpha: &[u8], rng: &mut R) -> Vec<u8> {
    let mut edits = vec![0u8; a.len()];
    let curr_k: usize = rng.gen_range(k / 2..k + 1);
    let mut idx: Vec<usize> = (0usize..a.len()).collect();
    idx.shuffle(rng);

    for i in 0..curr_k {
        edits[idx[i]] = rng.gen_range(1u8..4u8);
    }

    let mut b = vec![];

    for i in 0..a.len() {
        match edits[i] {
            0u8 => { // same
                b.push(a[i]);
            },
            1u8 => { // diff
                let mut iter = alpha.choose_multiple(rng, 2);
                let first = *iter.next().unwrap();
                let second = *iter.next().unwrap();
                b.push(if first == a[i] { second } else { first });
            },
            2u8 => { // insert
                b.push(*alpha.choose(rng).unwrap());
                b.push(a[i]);
            },
            3u8 => (), // delete
            _ => panic!("This should not have been reached!")
        }
    }

    b
}

pub fn rand_str<R: Rng>(length: usize, alpha: &[u8], rng: &mut R) -> Vec<u8> {
    let mut res = vec![0u8; length];

    for i in 0..length {
        res[i] = *alpha.choose(rng).unwrap();
    }

    res
}
