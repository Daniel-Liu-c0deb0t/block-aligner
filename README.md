# Better-Alignment
Better alignment algorithms.

You will probably need these programs on your `$PATH`:
* wasmtime
* binaryen wasm-opt
* wabt wasm2wat

## WASM SIMD support
* [ ] Use bitmask instruction instead of workaround
* [ ] Make sure functions without target feature attribute are inlined correctly.
Easy fix: run binaryen wasm-opt pass with lots of inlining.
* [ ] Too many `locals.get` and `locals.set` in generated output.
Hopefully, better compiler output in the future will fix this.
* [ ] Too many function calls in the x86 assembly generated from the optimized WASM.
* [x] Try wasmer and wavm. They don't offer much performance improvement over wasmtime.
