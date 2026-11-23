#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct __attribute__((packed)) { char id[32]; uint32_t tokens; } record_t;
typedef struct { int fd; size_t n; record_t *rows; size_t i; } stream_t;

static void open_stream(stream_t *s, const char *path) {
    struct stat st;
    s->fd = open(path, O_RDONLY);
    if (s->fd < 0 || fstat(s->fd, &st) != 0 || st.st_size % sizeof(record_t) != 0) {
        fprintf(stderr, "invalid record file %s: %s\n", path, strerror(errno)); exit(2);
    }
    s->n = (size_t)st.st_size / sizeof(record_t); s->i = 0;
    s->rows = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, s->fd, 0);
    if (s->rows == MAP_FAILED) { fprintf(stderr, "mmap failed %s\n", path); exit(2); }
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: overlap_merge warm stable anneal\n"); return 2; }
    stream_t s[3]; for (int k=0;k<3;k++) open_stream(&s[k], argv[k+1]);
    uint64_t unique_docs=0, unique_tokens=0, occurrence_docs=0, occurrence_tokens=0;
    uint64_t pair_docs[3]={0}, pair_tokens[3]={0}, triple_docs=0, triple_tokens=0, mismatches=0;
    while (s[0].i<s[0].n || s[1].i<s[1].n || s[2].i<s[2].n) {
        int m=-1;
        for (int k=0;k<3;k++) if (s[k].i<s[k].n && (m<0 || memcmp(s[k].rows[s[k].i].id,s[m].rows[s[m].i].id,32)<0)) m=k;
        char minimum_id[32]; memcpy(minimum_id, s[m].rows[s[m].i].id, 32);
        int mask=0; uint32_t max_tokens=0, first_tokens=0;
        for (int k=0;k<3;k++) if (s[k].i<s[k].n && memcmp(s[k].rows[s[k].i].id,minimum_id,32)==0) {
            uint32_t t=s[k].rows[s[k].i].tokens; mask|=1<<k; occurrence_docs++; occurrence_tokens+=t;
            if (!first_tokens) first_tokens=t; else if (first_tokens!=t) mismatches++;
            if (t>max_tokens) max_tokens=t; s[k].i++;
        }
        unique_docs++; unique_tokens+=max_tokens;
        if ((mask&3)==3) { pair_docs[0]++; pair_tokens[0]+=max_tokens; }
        if ((mask&5)==5) { pair_docs[1]++; pair_tokens[1]+=max_tokens; }
        if ((mask&6)==6) { pair_docs[2]++; pair_tokens[2]+=max_tokens; }
        if (mask==7) { triple_docs++; triple_tokens+=max_tokens; }
    }
    printf("{\"unique_docs\":%llu,\"unique_reference_tokens\":%llu,"
           "\"occurrence_docs\":%llu,\"occurrence_reference_tokens\":%llu,"
           "\"warmup_stable_docs\":%llu,\"warmup_stable_tokens\":%llu,"
           "\"warmup_anneal_docs\":%llu,\"warmup_anneal_tokens\":%llu,"
           "\"stable_anneal_docs\":%llu,\"stable_anneal_tokens\":%llu,"
           "\"triple_docs\":%llu,\"triple_tokens\":%llu,\"token_mismatches\":%llu}\n",
           (unsigned long long)unique_docs,(unsigned long long)unique_tokens,
           (unsigned long long)occurrence_docs,(unsigned long long)occurrence_tokens,
           (unsigned long long)pair_docs[0],(unsigned long long)pair_tokens[0],
           (unsigned long long)pair_docs[1],(unsigned long long)pair_tokens[1],
           (unsigned long long)pair_docs[2],(unsigned long long)pair_tokens[2],
           (unsigned long long)triple_docs,(unsigned long long)triple_tokens,
           (unsigned long long)mismatches);
    for(int k=0;k<3;k++){ munmap(s[k].rows,s[k].n*sizeof(record_t)); close(s[k].fd); }
    return 0;
}
