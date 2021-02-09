cargo clean
RUSTFLAGS="--emit llvm-ir,asm -C llvm-args=-x86-asm-syntax=intel" cargo build --release --benches
