CC=clang
CFLAGS=-L../target/release -lblock_aligner -I.

.PHONY: all block_aligner example

all: example

block_aligner:
	cargo build --release --features c_bindings,simd_avx2

example: block_aligner example.c
	$(CC) $(CFLAGS) example.c
