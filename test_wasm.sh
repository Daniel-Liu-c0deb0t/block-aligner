CARGO_TARGET_WASM32_WASI_RUNNER="wasmtime --wasm-features simd --" RUSTFLAGS="-C target-feature=+simd128" cargo test --target=wasm32-wasi --all-targets -- --nocapture "$@"
