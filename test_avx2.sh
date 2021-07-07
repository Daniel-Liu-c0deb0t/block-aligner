RUSTFLAGS="-C target-cpu=native" cargo test --all-targets -- "$@"
RUSTFLAGS="-C target-cpu=native" RUSTDOCFLAGS="-C target-cpu=native" cargo test --doc -- "$@"
