#include <stdio.h>

#include "block_aligner.h"

void example1(void) {
    // global alignment
    const char* a_str = "AAAAAAAA";
    const char* b_str = "AARAAAA";
    SizeRange range = {.min = 32, .max = 32};
    Gaps gaps = {.open = -11, .extend = -1};

    PaddedBytes* a = block_make_padded_aa(a_str, range.max);
    PaddedBytes* b = block_make_padded_aa(b_str, range.max);

    BlockHandle block = block_align_aa(a, b, &BLOSUM62, gaps, range);
    AlignResult res = block_res_aa(block);

    printf("a: %s\nb: %s\nscore: %d\nidx: (%lu, %lu)\n",
            a_str,
            b_str,
            res.score,
            res.query_idx,
            res.reference_idx);

    block_free_aa(block);
    block_free_padded_aa(a);
    block_free_padded_aa(b);
}

void example2(void) {
    // global alignment with traceback
    const char* a_str = "AAAAAAAA";
    const char* b_str = "AARAAAA";
    SizeRange range = {.min = 32, .max = 32};
    Gaps gaps = {.open = -11, .extend = -1};

    PaddedBytes* a = block_make_padded_aa(a_str, range.max);
    PaddedBytes* b = block_make_padded_aa(b_str, range.max);

    BlockHandle block = block_align_aa_trace(a, b, &BLOSUM62, gaps, range);
    AlignResult res = block_res_aa_trace(block);

    printf("a: %s\nb: %s\nscore: %d\nidx: (%lu, %lu)\n",
            a_str,
            b_str,
            res.score,
            res.query_idx,
            res.reference_idx);

    CigarVec cigar = block_cigar_aa_trace(block);
    // Note: 'M' signals either a match or mismatch
    char ops_char[] = {' ', 'M', 'I', 'D'};
    for (int i = 0; i < cigar.len; i++) {
        printf("%lu%c", cigar.ptr[i].len, ops_char[cigar.ptr[i].op]);
    }
    printf("\n");

    block_free_cigar(cigar);
    block_free_aa_trace(block);
    block_free_padded_aa(a);
    block_free_padded_aa(b);
}

int main() {
    example1();
    example2();
}
