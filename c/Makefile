CC=clang
CFLAGS=-L../target/release -lblock_aligner -I.

.PHONY: all block_aligner example

all: example

block_aligner:
	cargo build --release --features simd_avx2
	cbindgen --config ../cbindgen.toml --crate block-aligner --output block_aligner.h --quiet ..

example: block_aligner example.c
	$(CC) $(CFLAGS) example.c
