CARGO_TARGET_WASM32_WASI_RUNNER="wasmtime --wasm-features simd --" RUSTFLAGS="-C target-feature=+simd128" cargo run --target=wasm32-wasi --example accuracy --release -- "$@"
