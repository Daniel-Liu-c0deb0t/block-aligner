set -e

cargo clean
RUSTFLAGS="-C target-feature=+simd128" cargo build --release --benches --target wasm32-wasi

# binaryen wasm-opt pass
for f in target/wasm32-wasi/*/deps/*.wasm; do
    # extreme inlining
    wasm-opt --enable-simd --enable-sign-ext -O4 --inlining-optimizing -ifwl -ocimfs 300 -fimfs 300 -aimfs 20 -o $f.opt $f
    echo $f.opt
done

# demangle symbols
cargo install rustfilt
for f in target/wasm32-wasi/*/deps/*.wasm.opt; do
    wasm2wat --enable-simd $f | rustfilt -o $f.wat
    echo "$f.wat"
done

# disassemble wasmtime generated object files with objdump
for f in target/wasm32-wasi/*/deps/*.wasm.opt; do
    wasmtime wasm2obj --enable-simd $f $f.o
    objdump -drwSl -x86-asm-syntax=intel $f.o | rustfilt -o $f.objdump
    echo "$f.objdump"
done
