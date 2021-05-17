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

static FILE_NAMES: [[&str; 7]; 2] = [
    [
        "data/uc30_30_40.m8",
        "data/uc30_40_50.m8",
        "data/uc30_50_60.m8",
        "data/uc30_60_70.m8",
        "data/uc30_70_80.m8",
        "data/uc30_80_90.m8",
        "data/uc30_90_100.m8"
    ],
    [
        "data/merged_clu_aln_30_40.m8",
        "data/merged_clu_aln_40_50.m8",
        "data/merged_clu_aln_50_60.m8",
        "data/merged_clu_aln_60_70.m8",
        "data/merged_clu_aln_70_80.m8",
        "data/merged_clu_aln_80_90.m8",
        "data/merged_clu_aln_90_100.m8"
    ]
];

fn get_data(file_names: &[&str]) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut res = vec![];

    for file_name in file_names {
        let reader = BufReader::new(File::open(file_name).unwrap());

        for line in reader.lines() {
            let line = line.unwrap();
            let mut last_two = line.split_ascii_whitespace().rev().take(2);
            let r = last_two.next().unwrap().to_ascii_uppercase();
            let q = last_two.next().unwrap().to_ascii_uppercase();

            res.push((q.as_bytes().to_owned(), r.as_bytes().to_owned()));
        }
    }

    res
}

#[cfg(not(target_arch = "wasm32"))]
fn bench_parasailors_aa_core(idx: usize) -> i32 {
    let file_data = get_data(&FILE_NAMES[idx]);
    let matrix = Matrix::new(MatrixType::Blosum62);
    let data = file_data
        .iter()
        .map(|(q, r)| (Profile::new(q, &matrix), r.to_owned()))
        .collect::<Vec<(Profile, Vec<u8>)>>();

    let mut temp = 0i32;
    for (p, r) in &data {
        temp = temp.wrapping_add(global_alignment_score(p, r, 11, 1));
    }
    temp
}

fn bench_scan_aa_core(idx: usize) -> i32 {
    let file_data = get_data(&FILE_NAMES[idx]);
    let data = file_data
        .iter()
        .map(|(q, r)| (PaddedBytes::from_bytes(q, 2048, false), PaddedBytes::from_bytes(r, 2048, false)))
        .collect::<Vec<(PaddedBytes, PaddedBytes)>>();
    type BenchParams = GapParams<-11, -1>;

    let mut temp = 0i32;
    for (q, r) in &data {
        let a = Block::<BenchParams, _, 32, 256, false, false>::align(&q, &r, &BLOSUM62, 0);
        temp = temp.wrapping_add(a.res().score); // prevent optimizations
    }
    temp
}

fn time(f: fn(usize) -> i32, idx: usize) -> Duration {
    let start = Instant::now();
    black_box(f(idx));
    start.elapsed()
}

fn main() {
    let d = time(bench_scan_aa_core, 0);
    println!("scan uc time (s): {}", d.as_secs_f64());
    let d = time(bench_scan_aa_core, 1);
    println!("scan merged time (s): {}", d.as_secs_f64());

    #[cfg(not(target_arch = "wasm32"))]
    {
        let d = time(bench_parasailors_aa_core, 0);
        println!("parasail uc time (s): {}", d.as_secs_f64());
        let d = time(bench_parasailors_aa_core, 1);
        println!("parasail merged time (s): {}", d.as_secs_f64());
    }
}
