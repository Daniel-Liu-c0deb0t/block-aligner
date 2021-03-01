RUSTFLAGS="-C target-feature=+simd128" CARGO_TARGET_WASM32_WASI_RUNNER="wasmtime --enable-simd --" cargo bench --target=wasm32-wasi
