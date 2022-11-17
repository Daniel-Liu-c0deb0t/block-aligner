#[cfg(not(feature = "simd_avx2"))]
fn main() {}

#[cfg(feature = "simd_avx2")]
fn test(file_name: &str, min_size: usize, max_size: usize, name: &str, verbose: bool, writer: &mut impl std::io::Write) -> (usize, f64, usize) {
    use parasailors::{Matrix, *};

    use bio::alignment::distance::simd::levenshtein;

    use block_aligner::scan_block::*;
    use block_aligner::scores::*;

    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let mut wrong = 0usize;
    let mut wrong_avg = 0f64;
    let mut count = 0usize;
    let reader = BufReader::new(File::open(file_name).unwrap());
    let all_lines = reader.lines().collect::<Vec<_>>();

    for lines in all_lines.chunks(2) {
        let r = lines[0].as_ref().unwrap().to_ascii_uppercase();
        let q = lines[1].as_ref().unwrap().to_ascii_uppercase();

        // parasail
        let matrix = Matrix::new(MatrixType::IdentityWithPenalty);
        let profile = parasailors::Profile::new(q.as_bytes(), &matrix);
        let parasail_score = global_alignment_score(&profile, r.as_bytes(), 2, 1);

        let r_padded = PaddedBytes::from_bytes::<NucMatrix>(r.as_bytes(), 2048);
        let q_padded = PaddedBytes::from_bytes::<NucMatrix>(q.as_bytes(), 2048);
        let run_gaps = Gaps { open: -2, extend: -1 };

        // ours
        let mut block_aligner = Block::<false, false>::new(q.len(), r.len(), max_size);
        block_aligner.align(&q_padded, &r_padded, &NW1, run_gaps, min_size..=max_size, 0);
        let scan_score = block_aligner.res().score;

        write!(
            writer,
            "{}, {}-{}, {}, {}, {}, {}\n",
            name,
            min_size,
            max_size,
            q.len(),
            r.len(),
            scan_score,
            parasail_score
        ).unwrap();

        if parasail_score != scan_score {
            wrong += 1;
            wrong_avg += ((parasail_score - scan_score) as f64) / (parasail_score as f64);

            if verbose {
                let edit_dist = levenshtein(q.as_bytes(), r.as_bytes());
                println!(
                    "parasail: {}, ours: {}, edit dist: {}\nq (len = {}): {}\nr (len = {}): {}",
                    parasail_score,
                    scan_score,
                    edit_dist,
                    q.len(),
                    q,
                    r.len(),
                    r
                );
            }
        }

        count += 1;
    }

    (wrong, wrong_avg / (wrong as f64), count)
}

#[cfg(feature = "simd_avx2")]
fn main() {
    use std::env;
    use std::fs::File;
    use std::io::{Write, BufWriter};

    let arg1 = env::args().skip(1).next();
    let verbose = arg1.is_some() && arg1.unwrap() == "-v";
    let paths = ["data/real.illumina.b10M.txt", "data/real.ont.b10M.txt", "data/sequences.txt"];
    let names = ["illumina", "nanopore 1kbp", "nanopore 25kbp"];
    let min_size = [32, 32, 32];
    let max_size = [32, 128, 256];

    let out_file_name = "data/nanopore_accuracy.csv";
    let mut writer = BufWriter::new(File::create(out_file_name).unwrap());
    write!(writer, "dataset, size, query len, reference len, pred score, true score\n").unwrap();

    println!("\ndataset, size, total, wrong, wrong % error");

    for ((path, name), (&min_size, &max_size)) in paths.iter().zip(&names).zip(min_size.iter().zip(&max_size)) {
        let (wrong, wrong_avg, count) = test(path, min_size, max_size, name, verbose, &mut writer);
        println!("\n{}, {}-{}, {}, {}, {}", name, min_size, max_size, count, wrong, wrong_avg);
    }

    println!("# Done!");
}
