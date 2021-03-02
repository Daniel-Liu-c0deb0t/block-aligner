cargo clean

CARGO_TARGET_WASM32_WASI_RUNNER="wasmtime --enable-simd --"
# CARGO_TARGET_WASM32_WASI_RUNNER="wasmer --native --llvm --enable-simd --"

RUSTFLAGS="-C target-feature=+simd128" cargo build --release --benches --target=wasm32-wasi

# binaryen wasm-opt pass
for f in target/wasm32-wasi/*/deps/*.wasm; do
    wasm-opt --enable-simd -O4 --inlining-optimizing -ifwl -ocimfs 300 -fimfs 300 -aimfs 20 -o $f.opt $f
    echo $f.opt
done

for f in target/wasm32-wasi/*/deps/*.wasm.opt; do
    $CARGO_TARGET_WASM32_WASI_RUNNER $f --bench
done
