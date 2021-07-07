RUSTFLAGS="-C target-cpu=native" cargo test --all-targets -- "$@"
RUSTFLAGS="-C target-cpu=native" cargo test --doc -- "$@"
