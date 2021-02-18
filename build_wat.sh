cargo clean
RUSTFLAGS="-g --target=wasm32-wasi -C target-feature=+simd128" cargo build --release --benches

# demangle symbols
cargo install rustfilt
for f in target/*/deps/*.wasm; do
    wasm2wat --enable-simd $f | rustfilt -o $f.wat
    echo "$f.wat"
done
