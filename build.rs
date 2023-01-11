fn main() {
    #[cfg(feature = "cbindgen")]
    {
        if std::env::var("BLOCK_ALIGNER_C").is_ok() {
            let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
            cbindgen::generate(&crate_dir)
                .unwrap()
                .write_to_file(format!("{}/c/block_aligner.h", crate_dir));
        }
    }
}
