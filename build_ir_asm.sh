cargo clean
# make sure debug info is generated
RUSTFLAGS="-g --emit llvm-ir,asm -C llvm-args=-x86-asm-syntax=intel -C target-cpu=native" cargo build --release --benches

# demangle symbols
cargo install rustfilt
for f in target/*/deps/*.{s,ll}; do
    rustfilt -i $f -o $f.new
    mv $f.new $f
    echo $f
done

# also create source/asm interleaved version with objdump
shopt -s extglob
for f in target/*/deps/!(*.*); do
    objdump -drwSl -x86-asm-syntax=intel $f | rustfilt -o $f.objdump
    echo "$f.objdump"
done
