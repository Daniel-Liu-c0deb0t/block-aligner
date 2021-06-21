#include <stdio.h>

#include "block_aligner.h"

int main() {
    const char* a_str = "AAAAAAAA";
    const char* b_str = "AARAAAA";
    SizeRange range = {.min = 32, .max = 32};
    Gaps gaps = {.open = -11, .extend = -1};

    PaddedBytes* a = block_make_padded_aa(a_str, range.max);
    PaddedBytes* b = block_make_padded_aa(b_str, range.max);

    BlockHandle block = block_align_aa(a, b, &BLOSUM62, gaps, range);
    AlignResult res = block_res_aa(block);

    printf("a: %s\nb: %s\nscore: %d\nidx: (%lu, %lu)\n", a_str, b_str, res.score, res.query_idx, res.reference_idx);

    block_free_aa(block);
}
