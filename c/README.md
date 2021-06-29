# C API
This directory contains an example of how to use the C API of block aligner.

Currently, only protein alignment is supported with the C API. Other features
may be added if there is demand for them.

## Running the example
1. `cd` into this directory.
2. Run `make`. This will build block aligner in release mode, use cbindgen
to generate the header file, and make sure block aligner is linked to the
example program.
3. Run `./a.out`. This will run the example program to perform alignment
calculations.

The generated header file, `c/block_aligner.h`, should be included in
code that calls block aligner functions. It should be C++ compatible.
