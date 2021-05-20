#![feature(bench_black_box)]
#![cfg(any(
        all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"),
        all(target_arch = "wasm32", target_feature = "simd128")
))]

#[cfg(not(target_arch = "wasm32"))]
use parasailors::{Matrix, *};

use rand::prelude::*;

use block_aligner::scan_block::*;
use block_aligner::scores::*;

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::{Instant, Duration};
use std::hint::black_box;
use std::iter;

use block_aligner::simulate::*;

static FILE_NAME: &str = "data/supplementary_data/sequences.txt";
const ITER: usize = 10000;
const LEN: usize = 10000;
const K: usize = 1000;

fn get_data(file_name: Option<&str>) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut rng = StdRng::seed_from_u64(1234);

    if let Some(file_name) = file_name {
        let mut res = vec![];

        let reader = BufReader::new(File::open(file_name).unwrap());
        let all_lines = reader.lines().collect::<Vec<_>>();

        for lines in all_lines.chunks(2) {
            let r = lines[0].as_ref().unwrap().to_ascii_uppercase();
            let q = lines[1].as_ref().unwrap().to_ascii_uppercase();
            let mut r = r.as_bytes().to_owned();
            let mut q = q.as_bytes().to_owned();
            let extend_r = rand_str(100, &NUC, &mut rng);
            let extend_q = rand_str(100, &NUC, &mut rng);
            r.extend_from_slice(&extend_r);
            q.extend_from_slice(&extend_q);
            res.push((q, r));
        }

        res
    } else {
        let mut r = rand_str(LEN, &NUC, &mut rng);
        let mut q = rand_mutate(&r, K, &NUC, &mut rng);
        let extend_r = rand_str(500, &NUC, &mut rng);
        let extend_q = rand_str(500, &NUC, &mut rng);
        r.extend_from_slice(&extend_r);
        q.extend_from_slice(&extend_q);
        black_box(iter::repeat_with(|| (q.clone(), r.clone())).take(ITER).collect())
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn bench_parasailors_nuc_core(file: bool) -> (i32, Duration) {
    let file_data = get_data(if file { Some(&FILE_NAME) } else { None });
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

fn bench_scan_nuc_core(file: bool) -> (i32, Duration) {
    let file_data = get_data(if file { Some(&FILE_NAME) } else { None });
    let x_drop = if file { 50 } else { 100 };
    let data = file_data
        .iter()
        .map(|(q, r)| (PaddedBytes::from_bytes(q, 2048, true), PaddedBytes::from_bytes(r, 2048, true)))
        .collect::<Vec<(PaddedBytes, PaddedBytes)>>();
    type BenchParams = GapParams<-2, -1>;

    let start = Instant::now();
    let mut temp = 0i32;
    for (q, r) in &data {
        let a = Block::<BenchParams, _, false, true>::align(&q, &r, &NW1, 32..=32, x_drop);
        temp = temp.wrapping_add(a.res().score); // prevent optimizations
    }
    (temp, start.elapsed())
}

fn time(f: fn(bool) -> (i32, Duration), file: bool) -> Duration {
    let (temp, duration) = f(file);
    black_box(temp);
    duration
}

fn main() {
    let d = time(bench_scan_nuc_core, true);
    println!("scan nanopore time (s): {}", d.as_secs_f64());
    let d = time(bench_scan_nuc_core, false);
    println!("scan rand time (s): {}", d.as_secs_f64());

    #[cfg(not(target_arch = "wasm32"))]
    {
        let d = time(bench_parasailors_nuc_core, true);
        println!("parasail nanopore time (s): {}", d.as_secs_f64());
        let d = time(bench_parasailors_nuc_core, false);
        println!("parasail rand time (s): {}", d.as_secs_f64());
    }
}
