#![feature(bench_black_box)]

#[cfg(not(any(feature = "simd_wasm", feature = "simd_neon")))]
use parasailors::{Matrix, *};

#[cfg(not(any(feature = "simd_wasm", feature = "simd_neon")))]
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

#[cfg(not(any(feature = "simd_wasm", feature = "simd_neon")))]
fn bench_wfa2(file: &str, use_heuristic: bool) -> f64 {
    let data = get_data(file);

    let mut total_time = 0f64;
    let mut temp = 0i32;
    for (q, r) in &data {
        let mut wfa = WFAlignerGapAffine::new(1, 1, 1, AlignmentScope::Score, MemoryModel::MemoryHigh);
        if use_heuristic {
            wfa.set_heuristic(Heuristic::WFadaptive(10, 50, 1));
        } else {
            wfa.set_heuristic(Heuristic::None);
        }
        let start = Instant::now();
        wfa.align_end_to_end(&q, &r);
        total_time += start.elapsed().as_secs_f64();
        temp = temp.wrapping_add(wfa.score());
    }
    black_box(temp);
    total_time
}

fn bench_ours(file: &str, trace: bool, size: (usize, usize)) -> f64 {
    let file_data = get_data(file);
    let data = file_data
        .iter()
        .map(|(q, r)| (PaddedBytes::from_bytes::<NucMatrix>(q, size.1), PaddedBytes::from_bytes::<NucMatrix>(r, size.1)))
        .collect::<Vec<(PaddedBytes, PaddedBytes)>>();
    let bench_gaps = Gaps { open: -2, extend: -1 };

    let mut total_time = 0f64;
    let mut temp = 0i32;
    for (q, r) in &data {
        if trace {
            let mut a = Block::<true, false>::new(q.len(), r.len(), size.1);
            let start = Instant::now();
            a.align(&q, &r, &NW1, bench_gaps, size.0..=size.1, 0);
            total_time += start.elapsed().as_secs_f64();
            temp = temp.wrapping_add(a.res().score); // prevent optimizations
        } else {
            let mut a = Block::<false, false>::new(q.len(), r.len(), size.1);
            let start = Instant::now();
            a.align(&q, &r, &NW1, bench_gaps, size.0..=size.1, 0);
            total_time += start.elapsed().as_secs_f64();
            temp = temp.wrapping_add(a.res().score); // prevent optimizations
        }
    }
    black_box(temp);
    total_time
}

fn main() {
    let files = ["data/real.illumina.b10M.txt", "data/real.ont.b10M.txt", "data/seq_pairs.10kbps.5000.txt", "data/seq_pairs.50kbps.10000.txt"];
    let names = ["illumina", "nanopore 1kbp", "nanopore <10kbp", "nanopore <50kbp"];
    let sizes = [[(32, 32), (32, 32)], [(32, 32), (32, 128)], [(128, 128), (128, 1024)], [(512, 512), (512, 4096)]];
    let run_parasail_arr = [true, true, true, false];

    println!("# time (s)");
    println!("dataset, algorithm, time");

    for (((file, name), size), &run_parasail) in files.iter().zip(&names).zip(&sizes).zip(&run_parasail_arr) {
        for &s in size {
            let t = bench_ours(file, false, s);
            println!("{}, ours ({}-{}), {}", name, s.0, s.1, t);
        }

        #[cfg(not(any(feature = "simd_wasm", feature = "simd_neon")))]
        {
            let t = bench_wfa2(file, false);
            println!("{}, wfa2, {}", name, t);

            let t = bench_wfa2(file, true);
            println!("{}, wfa2 adaptive, {}", name, t);
        }

        if run_parasail {
            #[cfg(not(any(feature = "simd_wasm", feature = "simd_neon")))]
            {
                let t = bench_parasailors(file);
                println!("{}, parasail, {}", name, t);
            }
        }
    }
}
