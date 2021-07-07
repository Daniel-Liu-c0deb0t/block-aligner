# block aligner
[![CI](https://github.com/Daniel-Liu-c0deb0t/block-aligner/actions/workflows/ci.yaml/badge.svg)](https://github.com/Daniel-Liu-c0deb0t/block-aligner/actions/workflows/ci.yaml)
[![License](https://img.shields.io/github/license/Daniel-Liu-c0deb0t/block-aligner)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/block-aligner)](https://crates.io/crates/block_aligner)
[![Docs.rs](https://docs.rs/block-aligner/badge.svg)](https://docs.rs/block-aligner)

SIMD-accelerated library for computing global and X-drop affine gap sequence alignments using
an adaptive block-based algorithm.

## Example
```rust
use block_aligner::scan_block::*;
use block_aligner::scores::*;
use block_aligner::cigar::*;

let block_size = 16;
let gaps = Gaps { open: -2, extend: -1 };
let r = PaddedBytes::from_bytes::<NucMatrix>(b"TTAAAAAAATTTTTTTTTTTT", block_size);
let q = PaddedBytes::from_bytes::<NucMatrix>(b"TTTTTTTTAAAAAAATTTTTTTTT", block_size);

// Align with traceback, but no x drop threshold.
let a = Block::<_, true, false>::align(&q, &r, &NW1, gaps, block_size..=block_size, 0);
let res = a.res();

assert_eq!(res, AlignResult { score: 7, query_idx: 24, reference_idx: 21 });
assert_eq!(a.trace().cigar(res.query_idx, res.reference_idx).to_string(), "2M6I16M3D");
```

## Algorithm
Pairwise alignment (weighted edit distance) involves computing the scores for each cell of a
2D dynamic programming matrix to find out how two strings can optimally align.
However, often it is possible to obtain accurate alignment scores without computing
the entire DP matrix, through banding or other means.

Block aligner provides a new efficient way to compute alignments on proteins, DNA sequences,
and byte strings.
Scores are calculated in a small square block that is shifted down or right in a greedy
manner, based on the scores at the edges of the block.
This dynamic approach results in a much smaller calculated block area, at the expense of
some accuracy.
To address problems with handling large gaps, we detect gaps by keeping track of the number
of iterations without seeing score increases. We call this "Y-drop", where Y is the threshold
number of iterations.
When the Y-drop condition is met, the block goes "back in time" to the previous best
checkpoint, and the size of the block dynamically increases to attempt to span the large gap.

Block aligner is built to exploit SIMD parallelism on modern CPUs.
Currently, AVX2 (256-bit vectors) and WASM SIMD (128-bit vectors) are supported.
For score calculations, 16-bit score values (lanes) and 32-bit per block offsets are used.

## Install
This library requires the nightly Rust channel.

You can directly clone this repo, or add the following to your `Cargo.toml`:
```
[dependencies]
block-aligner = "*"
```
Your computer needs to have a CPU that supports AVX2.

When building your code that uses this library, it is important to specify the
correct flags to turn on specific target features that this library uses.

For x86 AVX2:
```
RUSTFLAGS="-C target-cpu=native" cargo build --release
```
or
```
RUSTFLAGS="-C target-feature=+avx2" cargo build --release
```

For WASM SIMD:
```
RUSTFLAGS="-C target-feature=+simd128" cargo build --target=wasm32-wasi --release
```

## Data
Some Nanopore (DNA) and Uniclust30 (protein) data are used in some tests and benchmarks.
You will need to download them by following the instructions in the [data readme](data/README.md).

## Test
1. `./test_avx2.sh` or `./test_wasm.sh`

CI will run these tests when commits are pushed to this repo.

For assessing the accuracy of block aligner on random data, run `./accuracy_avx2.sh`,
`./x_drop_accuracy_avx2.sh`, or `./accuracy_wasm.sh`.
For Nanopore or Uniclust30 data, run `./nanopore_accuracy.sh` or `./uc_accuracy.sh`.

For debugging, there exists a `debug` feature flag that prints out a lot of
useful info about the internal state of the aligner while it runs.
There is another feature flag, `debug_size`, that prints the sizes of blocks after they grow.
To manually inspect alignments, run `./debug_avx2.sh` with two sequences as arguments.

## Docs
1. `./doc_avx2.sh` or `./doc_wasm.sh`

This will build the docs locally.

## Compare
Edits were made to [Hajime Suzuki](https://github.com/ocxtal)'s adaptive banding benchmark code
and difference recurrence benchmark code. These edits are available [here](https://github.com/Daniel-Liu-c0deb0t/adaptivebandbench)
and [here](https://github.com/Daniel-Liu-c0deb0t/diff-bench-paper), respectively.
Go to those repos, then follow the instructions for installing and running the code.

If you run the scripts in those repos for comparing scores produced by different algorithms,
you should get `.tsv` generated files. Then, in this repo's directory, run
```
./compare.sh /path/to/file.tsv
```
to get the comparisons.

## Benchmark
1. `./bench_avx2.sh` or `./bench_wasm.sh`

For benchmarking Nanopore or Uniclust30 data, run `./nanopore_bench.sh` or `./uc_bench.sh`.

## Profiling with MacOS Instruments
Use
```
brew install cargo-instruments
RUSTFLAGS="-g -C target-cpu=native" cargo instruments --example profile --release --open
```

## Analyzing performance with LLVM-MCA
Use
```
./build_ir_asm.sh
```
to generate assembly output and run LLVM-MCA.

## Viewing the assembly
Use either `./build_ir_asm.sh`, `objdump -d` on a binary (avoids recompiling code in
some cases), or a more advanced tool like Ghidra (has a decompiler, too).

## WASM SIMD support
WASM SIMD has been stabilizing in Rust recently, so WASM support should be fairly good.
To run WASM programs, you will need [`wasmtime`](https://github.com/bytecodealliance/wasmtime)
installed and on your `$PATH`.

## C API
There are C bindings for block aligner. More information on how to use them is located in
the [C readme](c/README.md).

## Other SIMD instruction sets
* [ ] SSE4.1 (Depends on demand)
* [ ] AVX-512 (I don't have a machine to test)
* [ ] NEON (I don't have a machine to test)

## Some Failed Ideas
1. What if we took Daily's prefix scan idea and made it faster and made it banded using
ring buffers and had tons of 32-bit offsets for intervals of the band to prevent overflow?
(This actually works, but it is soooooo complex.)
2. What if we took that banded idea (a single thin vertical band) and made it adaptive?
3. What if we placed blocks like Minecraft, where there is no overlap between blocks?
4. What if we compared the rightmost column and bottommost row in each block to decide
which direction to shift? (Surprisingly, using the first couple of values in each column
or row is better than using the whole column/row. Also, comparing the sum of scores worked
better than comparing the max.)
5. Use a branch-predictor-like scheme to predict which direction to shift as a tie-breaker
when shifting right or down seem equally good.
6. ...
