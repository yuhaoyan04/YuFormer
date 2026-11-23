#define _GNU_SOURCE
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct __attribute__((packed)) {
    unsigned char doc_id[32];
    uint32_t doc_tokens;
    uint32_t begin;
    uint32_t end;
} record_t;

typedef struct { FILE *file; record_t row; int live; } input_t;

static void die(const char *message) {
    fprintf(stderr, "%s: %s\n", message, strerror(errno));
    exit(2);
}

static void advance(input_t *input) {
    size_t n = fread(&input->row, sizeof(record_t), 1, input->file);
    if (n == 1) {
        if (input->row.begin >= input->row.end || input->row.end > input->row.doc_tokens) {
            fprintf(stderr, "invalid consumed-token segment\n"); exit(2);
        }
        input->live = 1;
    } else {
        if (ferror(input->file)) die("read record");
        input->live = 0;
    }
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: overlap_consumed_merge warmup stable anneal\n"); return 2;
    }
    input_t in[3] = {0};
    for (int i = 0; i < 3; i++) {
        in[i].file = fopen(argv[i + 1], "rb"); if (!in[i].file) die(argv[i + 1]);
        advance(&in[i]);
    }
    uint64_t occurrence_docs = 0, unique_docs = 0, pair_docs = 0, triple_docs = 0;
    uint64_t occurrence_tokens = 0, unique_tokens = 0, token_mismatches = 0;
    while (in[0].live || in[1].live || in[2].live) {
        int first = -1;
        for (int i = 0; i < 3; i++)
            if (in[i].live && (first < 0 || memcmp(in[i].row.doc_id, in[first].row.doc_id, 32) < 0)) first = i;
        unsigned char target_id[32]; memcpy(target_id, in[first].row.doc_id, 32);
        record_t rows[3]; int count = 0;
        for (int i = 0; i < 3; i++) {
            if (in[i].live && memcmp(in[i].row.doc_id, target_id, 32) == 0) {
                rows[count++] = in[i].row; advance(&in[i]);
            }
        }
        unique_docs++; occurrence_docs += (uint64_t)count;
        if (count == 2) pair_docs++; else if (count == 3) triple_docs++;
        for (int i = 0; i < count; i++) {
            occurrence_tokens += (uint64_t)(rows[i].end - rows[i].begin);
            if (rows[i].doc_tokens != rows[0].doc_tokens) token_mismatches++;
        }
        for (int i = 0; i < count; i++) {
            for (int j = i + 1; j < count; j++) {
                if (rows[j].begin < rows[i].begin) { record_t t = rows[i]; rows[i] = rows[j]; rows[j] = t; }
            }
        }
        uint32_t begin = rows[0].begin, end = rows[0].end;
        for (int i = 1; i < count; i++) {
            if (rows[i].begin > end) {
                unique_tokens += (uint64_t)(end - begin);
                begin = rows[i].begin; end = rows[i].end;
            } else if (rows[i].end > end) end = rows[i].end;
        }
        unique_tokens += (uint64_t)(end - begin);
    }
    for (int i = 0; i < 3; i++) fclose(in[i].file);
    printf("{\"occurrence_docs\":%llu,\"unique_docs\":%llu,\"pair_docs\":%llu,"
           "\"triple_docs\":%llu,\"occurrence_consumed_tokens\":%llu,"
           "\"unique_consumed_tokens\":%llu,\"token_mismatches\":%llu}\n",
           (unsigned long long)occurrence_docs, (unsigned long long)unique_docs,
           (unsigned long long)pair_docs, (unsigned long long)triple_docs,
           (unsigned long long)occurrence_tokens, (unsigned long long)unique_tokens,
           (unsigned long long)token_mismatches);
    return 0;
}
