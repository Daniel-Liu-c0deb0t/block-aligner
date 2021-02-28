cargo clean
RUSTFLAGS="--target=wasm32-wasi -C target-feature=+simd128" cargo build --tests

for f in target/*/deps/*.wasm; do
    wasmtime --enable-simd -- $f --nocapture
done
