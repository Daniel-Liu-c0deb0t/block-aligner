#![cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"))]

use cbindgen;
use std::env;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    cbindgen::generate(&crate_dir)
        .unwrap()
        .write_to_file(format!("{}/c/block_aligner.h", crate_dir));
}
