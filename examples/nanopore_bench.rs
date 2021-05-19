#![feature(bench_black_box)]
#![cfg(any(
        all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"),
        all(target_arch = "wasm32", target_feature = "simd128")
))]

#[cfg(not(target_arch = "wasm32"))]
use parasailors::{Matrix, *};

use block_aligner::scan_block::*;
use block_aligner::scores::*;

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::{Instant, Duration};
use std::hint::black_box;

static FILE_NAME: &str = "data/supplementary_data/sequences.txt";

fn get_data(file_name: &str) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut res = vec![];

    let reader = BufReader::new(File::open(file_name).unwrap());
    let all_lines = reader.lines().collect::<Vec<_>>();

    for lines in all_lines.chunks(2) {
        let r = lines[0].as_ref().unwrap().to_ascii_uppercase();
        let q = lines[1].as_ref().unwrap().to_ascii_uppercase();
        res.push((q.as_bytes().to_owned(), r.as_bytes().to_owned()));
    }

    res
}

#[cfg(not(target_arch = "wasm32"))]
fn bench_parasailors_nuc_core() -> (i32, Duration) {
    let file_data = get_data(&FILE_NAME);
    let matrix = Matrix::new(MatrixType::IdentityWithPenalty);
    let data = file_data
        .iter()
        .map(|(q, r)| (Profile::new(q, &matrix), r.to_owned()))
        .collect::<Vec<(Profile, Vec<u8>)>>();

    let start = Instant::now();
    let mut temp = 0i32;
    for (p, r) in &data {
        temp = temp.wrapping_add(global_alignment_score(p, r, 2, 1));
    }
    (temp, start.elapsed())
}

fn bench_scan_nuc_core() -> (i32, Duration) {
    let file_data = get_data(&FILE_NAME);
    let data = file_data
        .iter()
        .map(|(q, r)| (PaddedBytes::from_bytes(q, 2048, true), PaddedBytes::from_bytes(r, 2048, true)))
        .collect::<Vec<(PaddedBytes, PaddedBytes)>>();
    type BenchParams = GapParams<-2, -1>;

    let start = Instant::now();
    let mut temp = 0i32;
    for (q, r) in &data {
        let a = Block::<BenchParams, _, false, true>::align(&q, &r, &NW1, 32..=32, 50);
        temp = temp.wrapping_add(a.res().score); // prevent optimizations
    }
    (temp, start.elapsed())
}

fn time(f: fn() -> (i32, Duration)) -> Duration {
    let (temp, duration) = f();
    black_box(temp);
    duration
}

fn main() {
    let d = time(bench_scan_nuc_core);
    println!("scan time (s): {}", d.as_secs_f64());

    #[cfg(not(target_arch = "wasm32"))]
    {
        let d = time(bench_parasailors_nuc_core);
        println!("parasail time (s): {}", d.as_secs_f64());
    }
}
