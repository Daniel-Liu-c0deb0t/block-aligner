# block aligner
SIMD-accelerated library for computing global and X-drop affine gap sequence alignments using
an adaptive block-based algorithm.

*This is very much work in progress. It's not cleaned up and the API is not stable.*

Here is a quick overview of the general algorithm ideas:

Block aligner (tentative name) provides an efficient way to compute alignments on both
proteins and DNA sequences. This is done by calculating scores in a square block. Based
on the scores, this block is shifted down or right, in a greedy, gradient-ascent-like manner,
like Suzuki's previous adaptive banding algorithms.
To handle large gaps, we introduce something called "Y-drop" that works like X-drop to identify
gaps and grow the size of the block. When the Y-drop condition is met, the block goes
"back in time" to the previous best location, and the size of the computed block increases.

To make this calculation efficient, SIMD vectors are used. Currently, 16-bit lanes and 32-bit
offsets are implemented. Instead of the typical anti-diagonal
adaptive banding or striped profile to avoid annoying dependencies between DP cells, we go
with calculating contiguous cells in a row or column and use prefix scan to resolve dependencies.
In other words, the vector is placed horizontally or vertically in the DP matrix.
This simplifies implementation and makes it a LOT more natural to reason about. This was avoided
in previous works due to performance concerns, but with small, adaptive bands, the benefits of the more
complex striped profile would probably not show up. With horizontal and vertical vectors,
we can easily look up scores with shuffles for protein alignment, where there are 20 amino acids.
Of course, this is probably a bit slower than the anti-diagonal approach, but you also can't handle
protein scoring matrices or grow any blocks to handle large gaps with that approach.

Everyone loves numbers, so here are some preliminary AVX2 results below. Note that there are a
lot of non-Rust libraries that are not here that are very fast.
More work is needed to compare against those.
```
test bench_parasailors_aa_1000_10000 ... bench:  46,750,058 ns/iter (+/- 2,308,345)
test bench_parasailors_aa_100_1000   ... bench:     543,164 ns/iter (+/- 7,508)
test bench_parasailors_aa_10_100     ... bench:      17,038 ns/iter (+/- 397)
test bench_rustbio_aa_100_1000       ... bench:  13,621,661 ns/iter (+/- 552,297)
test bench_rustbio_aa_10_100         ... bench:     132,481 ns/iter (+/- 2,011)
test bench_scan_aa_1000_10000        ... bench:     254,802 ns/iter (+/- 14,302)
test bench_scan_aa_1000_10000_insert ... bench:   2,141,701 ns/iter (+/- 105,654)
test bench_scan_aa_1000_10000_small  ... bench:     189,990 ns/iter (+/- 3,752)
test bench_scan_aa_100_1000          ... bench:      26,087 ns/iter (+/- 1,186)
test bench_scan_aa_100_1000_insert   ... bench:      44,264 ns/iter (+/- 497)
test bench_scan_aa_100_1000_small    ... bench:      19,081 ns/iter (+/- 209)
test bench_scan_aa_10_100            ... bench:       3,288 ns/iter (+/- 91)
test bench_scan_aa_10_100_insert     ... bench:       3,500 ns/iter (+/- 152)
test bench_scan_aa_10_100_small      ... bench:       2,081 ns/iter (+/- 40)
test bench_scan_nuc_1000_10000       ... bench:     195,366 ns/iter (+/- 4,742)
test bench_scan_nuc_100_1000         ... bench:      20,053 ns/iter (+/- 1,684)
```
Rustbio benchmarks are simple scalar implementations from the rust-bio library.
Parasailors is Rust bindings for Parasail, which calls an implementation of Farrar's algorithm
that automatically detects supported vector extensions and automatically uses 16-bit ints if
8-bit ints overflow. The `scan_aa` or `scan_nuc` benchmarks are for our block-based implementation.
The two numbers (eg., `100_1000`) means that there are 100 random sub/ins/del mutations and random test
sequences have lengths of around 1000. Insert means that a contiguous random sequence that is 10%
of the length of the entire sequence is inserted somewhere randomly (eg., for length = 1000, there is
a random insert of length 100). Small means that the calculated block is fixed in size, with a side
length of 16.

Note that the accuracy is ~100% for no insert, and drops to >95% with large inserts.
This was measured from random sequences of lengths = 100, 1000, or 10000, insert lengths = 10% of
sequence lengths, and varying mutation rates from 10% to 50% of the sequence length.
The tradeoff for this high accuracy is speed, as when there are long insertions, the block
needs to grow a lot to handle the gap.
This should be very promising vs previous adaptive banding approaches.
Note that the task here is global alignment, which should be harder than X-drop because
the block needs to make it all the way to the end of both strings without getting off track.

Additionally, the core prefix scan algorithm has been improved tremendously versus the naive
implementation used in Parasail:
```
test bench_naive_prefix_scan ... bench:          10 ns/iter (+/- 0)
test bench_opt_prefix_scan   ... bench:           1 ns/iter (+/- 0)
```
The main idea was more parallelism and avoiding slow AVX2 lane-crossing operations.
It was vital to optimize this part because we don't use the striped profile
from Farrar's algorithm and Daily's prefix scan variant that brings the step of fixing
up scores out of the innermost loop. This means that we do a prefix scan in
*every iteration* of the hottest loop in the entire library :)

## Install
1. Clone this repo

It's not on crates.io yet.

## Test
1. `./test_avx2.sh` or `./test_wasm.sh`

For assessing the accuracy of the aligner, run `./accuracy_avx2.sh`.

For debugging, there exists a feature `debug` feature flag that prints out a lot of
useful info about the internal state of the aligner while it runs.

## Benchmark
1. `./bench_avx2.sh` or `./bench_wasm.sh`

## Profiling with MacOS Instruments
Use
```
brew install cargo-instruments
RUSTFLAGS="-g -C target-cpu=native" cargo instruments --example profile --release --open
```

## WASM SIMD support
WASM SIMD support is very buggy. On some nightly versions it works, on some it doesn't.
Hopefully this will get better as things stabilize.

You will probably need these programs on your `$PATH`:
* wasmtime
* binaryen wasm-opt
* wabt wasm2wat

* [ ] Use bitmask instruction instead of workaround.
* [x] Make sure functions without target feature attribute are inlined correctly.
Easy fix: run binaryen wasm-opt pass with lots of inlining (currently done in the benchmark
script).
* [x] Try wasmer and wavm. Result: they don't offer much performance improvement over wasmtime.

## A Little History of Failed Ideas
1. What if we took Daily's prefix scan idea and made it faster and made it banded using
ring buffers and had tons of 32-bit offsets for intervals of the band to prevent overflow?
(this actually works, but it is soooooo complex)
2. What if we took that banded idea (a single thin vertical band) and made it adaptive?
3. What if we placed blocks like Minecraft, where there is no overlap between blocks?
4. ...
