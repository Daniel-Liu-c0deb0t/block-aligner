cargo clean
RUSTFLAGS="-g -C target-feature=+simd128" cargo build --release --benches --target wasm32-wasi

# demangle symbols
cargo install rustfilt
for f in target/wasm32-wasi/*/deps/*.wasm; do
    wasm2wat --enable-simd $f | rustfilt -o $f.wat
    echo "$f.wat"
done
