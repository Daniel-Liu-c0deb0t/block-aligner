RUSTDOCFLAGS="-C target-feature=+simd128" cargo doc --target=wasm32-wasi --no-deps --open
