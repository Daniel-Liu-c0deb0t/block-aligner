RUSTFLAGS="-C target-feature=+simd128" CARGO_TARGET_WASM32_WASI_RUNNER="wasmtime --enable-simd --" cargo test --target=wasm32-wasi --all-targets
