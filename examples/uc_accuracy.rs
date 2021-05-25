#![cfg(any(
        all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"),
        all(target_arch = "wasm32", target_feature = "simd128")
))]

use bio::alignment::pairwise::*;
use bio::alignment::{Alignment, AlignmentOperation};
use bio::scores::blosum62;

use block_aligner::scan_block::*;
use block_aligner::scores::*;

use std::{env, cmp};
use std::fs::File;
use std::io::{BufRead, BufReader};

fn test(file_name: &str, verbose: bool, wrong_indels: &mut [usize], count_indels: &mut [usize], wrong: &mut [usize], wrong_avg: &mut [i64], count: &mut [usize]) {
    let reader = BufReader::new(File::open(file_name).unwrap());

    for line in reader.lines() {
        let line = line.unwrap();
        let mut last_two = line.split_ascii_whitespace().rev().take(2);
        let r = last_two.next().unwrap().to_ascii_uppercase();
        let q = last_two.next().unwrap().to_ascii_uppercase();

        // rust-bio
        let mut bio_aligner = Aligner::with_capacity(q.len(), r.len(), -10, -1, &blosum62);
        let bio_alignment = bio_aligner.global(q.as_bytes(), r.as_bytes());
        let bio_score = bio_alignment.score;
        let seq_identity = seq_id(&bio_alignment);
        let id_idx = cmp::min((seq_identity * 10.0) as usize, 9);
        let indels = indels(&bio_alignment, cmp::max(q.len(), r.len()));
        let indels_idx = cmp::min((indels * 10.0) as usize, 9);

        let r_padded = PaddedBytes::from_bytes(r.as_bytes(), 2048, &BLOSUM62);
        let q_padded = PaddedBytes::from_bytes(q.as_bytes(), 2048, &BLOSUM62);
        type RunParams = GapParams<-11, -1>;

        // ours
        let block_aligner = Block::<RunParams, _, true, false>::align(&q_padded, &r_padded, &BLOSUM62, 32..=256, 0);
        let scan_res = block_aligner.res();
        let scan_score = scan_res.score;

        if bio_score != scan_score {
            wrong[id_idx] += 1;
            wrong_avg[id_idx] += (bio_score - scan_score) as i64;
            wrong_indels[indels_idx] += 1;

            if verbose {
                let (a_pretty, b_pretty) = block_aligner.trace().cigar(scan_res.query_idx, scan_res.reference_idx).format(q.as_bytes(), r.as_bytes());
                println!(
                    "seq id: {}, max indel len: {}, bio: {}, ours: {}\nq (len = {}): {}\nr (len = {}): {}\nbio pretty:\n{}\nours pretty:\n{}\n{}",
                    seq_identity,
                    indels,
                    bio_score,
                    scan_score,
                    q.len(),
                    q,
                    r.len(),
                    r,
                    bio_alignment.pretty(q.as_bytes(), r.as_bytes()),
                    a_pretty,
                    b_pretty
                );
            }
        }

        count[id_idx] += 1;
        count_indels[indels_idx] += 1;
    }
}

fn indels(a: &Alignment, len: usize) -> f64 {
    let mut indels = 0;

    for &op in &a.operations {
        if op == AlignmentOperation::Ins
            || op == AlignmentOperation::Del {
            indels += 1;
        }
    }
    (indels as f64) / (len as f64)
}

// BLAST sequence identity
fn seq_id(a: &Alignment) -> f64 {
    let mut matches = 0;

    for &op in &a.operations {
        if op == AlignmentOperation::Match {
            matches += 1;
        }
    }

    (matches as f64) / (a.operations.len() as f64)
}

fn main() {
    let arg1 = env::args().skip(1).next();
    let verbose = arg1.is_some() && arg1.unwrap() == "-v";
    let file_names_arr = [
        [
            "data/merged_clu_aln_30_40.m8",
            "data/merged_clu_aln_40_50.m8",
            "data/merged_clu_aln_50_60.m8",
            "data/merged_clu_aln_60_70.m8",
            "data/merged_clu_aln_70_80.m8",
            "data/merged_clu_aln_80_90.m8",
            "data/merged_clu_aln_90_100.m8"
        ],
        [
            "data/merged_clu_aln_0.95_30_40.m8",
            "data/merged_clu_aln_0.95_40_50.m8",
            "data/merged_clu_aln_0.95_50_60.m8",
            "data/merged_clu_aln_0.95_60_70.m8",
            "data/merged_clu_aln_0.95_70_80.m8",
            "data/merged_clu_aln_0.95_80_90.m8",
            "data/merged_clu_aln_0.95_90_100.m8"
        ],
        [
            "data/uc30_30_40.m8",
            "data/uc30_40_50.m8",
            "data/uc30_50_60.m8",
            "data/uc30_60_70.m8",
            "data/uc30_70_80.m8",
            "data/uc30_80_90.m8",
            "data/uc30_90_100.m8"
        ]
    ];
    let strings = ["merged_clu_aln", "merged_clu_aln_0.95", "uc30"];

    for (file_names, string) in file_names_arr.iter().zip(&strings) {
        println!("\n{}", string);

        let mut wrong_indels = [0usize; 10];
        let mut count_indels = [0usize; 10];
        let mut wrong = [0usize; 10];
        let mut wrong_avg = [0i64; 10];
        let mut count = [0usize; 10];

        for file_name in file_names {
            test(file_name, verbose, &mut wrong_indels, &mut count_indels, &mut wrong, &mut wrong_avg, &mut count);
        }

        println!("Seq identity");

        for i in 0..10 {
            println!(
                "bin: {}-{}, count: {}, wrong: {}, wrong avg: {}",
                (i as f64) / 10.0,
                ((i as f64) + 1.0) / 10.0,
                count[i],
                wrong[i],
                (wrong_avg[i] as f64) / (wrong[i] as f64)
            );
        }

        println!("\nIndels");

        for i in 0..10 {
            println!(
                "bin: {}-{}, count: {}, wrong: {}",
                (i as f64) / 10.0,
                ((i as f64) + 1.0) / 10.0,
                count_indels[i],
                wrong_indels[i]
            );
        }

        println!("\ntotal: {}, wrong: {}", count.iter().sum::<usize>(), wrong.iter().sum::<usize>());
    }

    println!("Done!");
}
