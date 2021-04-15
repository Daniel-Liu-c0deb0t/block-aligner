# Better-Alignment
Better alignment algorithms.

## Profiling with MacOS Instruments
Use
```
brew install cargo-instruments
RUSTFLAGS="-C target-cpu=native" cargo instruments --example profile --release --open
```
Make sure `debug = true` is in `Cargo.toml`.

## WASM SIMD support
WASM SIMD support is kind of buggy. Hopefully it'll get better as it stabilizes.

You will probably need these programs on your `$PATH`:
* wasmtime
* binaryen wasm-opt
* wabt wasm2wat

* [ ] Use bitmask instruction instead of workaround
* [x] Make sure functions without target feature attribute are inlined correctly.
Easy fix: run binaryen wasm-opt pass with lots of inlining.
* [x] Try wasmer and wavm. Result: they don't offer much performance improvement over wasmtime.
* [ ] Wrong results with release mode
