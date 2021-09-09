CC=clang
CFLAGS=-L../target/release -lblock_aligner -I.

.PHONY: all block_aligner example

all: example

block_aligner:
	BLOCK_ALIGNER_C= cargo build --release --features simd_avx2

example: block_aligner example.c
	LD_LIBRARY_PATH=../target/release $(CC) $(CFLAGS) example.c
