CC=clang
CFLAGS=-L../target/release -lblock_aligner -I.

.PHONY: all block_aligner example

all: example

block_aligner:
	RUSTFLAGS="-C target-cpu=native" BLOCK_ALIGNER_C= cargo build --release

example: block_aligner example.c
	$(CC) $(CFLAGS) example.c
