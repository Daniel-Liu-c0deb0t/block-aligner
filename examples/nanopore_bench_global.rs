#![feature(bench_black_box)]

#[cfg(not(any(feature = "simd_wasm", feature = "simd_neon")))]
use parasailors::{Matrix, *};

#[cfg(not(any(feature = "simd_wasm")))]
use rust_wfa2::aligner::*;

use block_aligner::scan_block::*;
use block_aligner::scores::*;

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;
use std::hint::black_box;

fn get_data(file_name: &str) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut res = vec![];

    let reader = BufReader::new(File::open(file_name).unwrap());
    let all_lines = reader.lines().collect::<Vec<_>>();

    for lines in all_lines.chunks(2) {
        let r = lines[0].as_ref().unwrap().to_ascii_uppercase();
        let q = lines[1].as_ref().unwrap().to_ascii_uppercase();
        let r = r.as_bytes().to_owned();
        let q = q.as_bytes().to_owned();
        res.push((q, r));
    }

    res
}

#[cfg(not(any(feature = "simd_wasm", feature = "simd_neon")))]
fn bench_parasailors(file: &str) -> f64 {
    let file_data = get_data(file);
    let matrix = Matrix::new(MatrixType::IdentityWithPenalty);
    let data = file_data
        .iter()
        .map(|(q, r)| (parasailors::Profile::new(q, &matrix), r.to_owned()))
        .collect::<Vec<(parasailors::Profile, Vec<u8>)>>();

    let start = Instant::now();
    let mut temp = 0i32;
    for (p, r) in &data {
        temp = temp.wrapping_add(global_alignment_score(p, r, 2, 1));
    }
    black_box(temp);
    start.elapsed().as_secs_f64()
}

#[cfg(not(feature = "simd_wasm"))]
fn bench_wfa2(file: &str, use_heuristic: bool) -> f64 {
    let data = get_data(file);

    let start = Instant::now();
    let mut temp = 0i32;
    for (q, r) in &data {
        let mut wfa = WFAlignerGapAffine::new(1, 1, 1, AlignmentScope::Score, MemoryModel::MemoryHigh);
        if use_heuristic {
            wfa.set_heuristic(Heuristic::BandedAdaptive(-5, 5, 1));
        } else {
            wfa.set_heuristic(Heuristic::None);
        }
        wfa.align_end_to_end(&q, &r);
        temp = temp.wrapping_add(wfa.score());
    }
    black_box(temp);
    start.elapsed().as_secs_f64()
}

fn bench_ours(file: &str, trace: bool, max_size: usize) -> f64 {
    let file_data = get_data(file);
    let data = file_data
        .iter()
        .map(|(q, r)| (PaddedBytes::from_bytes::<NucMatrix>(q, 2048), PaddedBytes::from_bytes::<NucMatrix>(r, 2048)))
        .collect::<Vec<(PaddedBytes, PaddedBytes)>>();
    let bench_gaps = Gaps { open: -2, extend: -1 };

    let start = Instant::now();
    let mut temp = 0i32;
    for (q, r) in &data {
        if trace {
            let mut a = Block::<true, false>::new(q.len(), r.len(), max_size);
            a.align(&q, &r, &NW1, bench_gaps, 32..=max_size, 0);
            temp = temp.wrapping_add(a.res().score); // prevent optimizations
        } else {
            let mut a = Block::<false, false>::new(q.len(), r.len(), max_size);
            a.align(&q, &r, &NW1, bench_gaps, 32..=max_size, 0);
            temp = temp.wrapping_add(a.res().score); // prevent optimizations
        }
    }
    black_box(temp);
    start.elapsed().as_secs_f64()
}

fn main() {
    let files = ["data/real.ont.b10M.txt"];
    let names = ["nanopore 1kbp"];

    println!("# time (s)");
    println!("dataset, algorithm, time");

    for (file, name) in files.iter().zip(&names) {
        let _t = bench_ours(file, false, 32);

        let t = bench_ours(file, false, 32);
        println!("{}, ours (32-32), {}", name, t);

        let t = bench_ours(file, false, 128);
        println!("{}, ours (32-128), {}", name, t);

        #[cfg(not(feature = "simd_wasm"))]
        {
            let t = bench_wfa2(file, false);
            println!("{}, wfa2, {}", name, t);

            let t = bench_wfa2(file, true);
            println!("{}, wfa2 adaptive band, {}", name, t);
        }

        #[cfg(not(any(feature = "simd_wasm", feature = "simd_neon")))]
        {
            let t = bench_parasailors(file);
            println!("{}, parasail, {}", name, t);
        }
    }
}
