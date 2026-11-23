#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    int pfd, ofd, tfd;
    uint64_t *positions, *offsets;
    unsigned char *tokens;
    uint64_t docs, token_count, index;
    size_t token_bytes;
} part_t;

static void die(const char *message) { fprintf(stderr, "%s: %s\n", message, strerror(errno)); exit(2); }

static void *map_exact(const char *path, size_t expected, int *fd) {
    struct stat st;
    *fd = open(path, O_RDONLY);
    if (*fd < 0 || fstat(*fd, &st) != 0) die(path);
    if ((uint64_t)st.st_size != (uint64_t)expected) {
        fprintf(stderr, "size mismatch %s: %llu != %llu\n", path,
                (unsigned long long)st.st_size, (unsigned long long)expected); exit(2);
    }
    if (!expected) { close(*fd); *fd = -1; return NULL; }
    void *mapped = mmap(NULL, expected, PROT_READ, MAP_SHARED, *fd, 0);
    if (mapped == MAP_FAILED) die("mmap");
    close(*fd);
    *fd = -1;
    return mapped;
}

static void flush_buffer(int fd, unsigned char *buffer, size_t *used) {
    size_t done = 0;
    while (done < *used) {
        ssize_t n = write(fd, buffer + done, *used - done);
        if (n <= 0) die("write output");
        done += (size_t)n;
    }
    *used = 0;
}

int main(int argc, char **argv) {
    if (argc != 5 && argc != 6) {
        fprintf(stderr, "usage: merge_token_parts manifest output expected_docs token_width [doc_lengths]\n"); return 2;
    }
    uint64_t expected_docs = strtoull(argv[3], NULL, 10);
    unsigned width = (unsigned)strtoul(argv[4], NULL, 10);
    if (width != 2 && width != 4) { fprintf(stderr, "invalid token width\n"); return 2; }
    FILE *manifest = fopen(argv[1], "r"); if (!manifest) die(argv[1]);
    size_t capacity = 1024, count = 0; part_t *parts = calloc(capacity, sizeof(part_t));
    char *line = NULL; size_t line_cap = 0;
    while (getline(&line, &line_cap, manifest) >= 0) {
        char *save = NULL, *pp = strtok_r(line, "\t\n", &save), *op = strtok_r(NULL, "\t\n", &save);
        char *tp = strtok_r(NULL, "\t\n", &save), *ds = strtok_r(NULL, "\t\n", &save);
        char *ts = strtok_r(NULL, "\t\n", &save);
        if (!pp || !op || !tp || !ds || !ts) { fprintf(stderr, "invalid manifest line\n"); return 2; }
        if (count == capacity) { capacity *= 2; parts = realloc(parts, capacity * sizeof(part_t)); if (!parts) die("realloc parts"); }
        part_t *p = &parts[count++]; memset(p, 0, sizeof(*p));
        p->docs = strtoull(ds, NULL, 10); p->token_count = strtoull(ts, NULL, 10);
        p->token_bytes = (size_t)p->token_count * width;
        p->positions = map_exact(pp, (size_t)p->docs * 8, &p->pfd);
        p->offsets = map_exact(op, (size_t)(p->docs + 1) * 8, &p->ofd);
        p->tokens = map_exact(tp, p->token_bytes, &p->tfd);
        if (p->offsets[p->docs] != p->token_count) { fprintf(stderr, "terminal offset mismatch\n"); return 2; }
    }
    free(line); fclose(manifest);
    if (count >= UINT32_MAX || expected_docs > SIZE_MAX / sizeof(uint32_t)) {
        fprintf(stderr, "part/document count exceeds position-map representation\n"); return 2;
    }
    size_t map_bytes = (size_t)expected_docs * sizeof(uint32_t);
    size_t mapped_bytes = map_bytes ? map_bytes : 1;
    uint32_t *part_for_position = mmap(NULL, mapped_bytes, PROT_READ | PROT_WRITE,
                                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (part_for_position == MAP_FAILED) die("position map mmap");
    memset(part_for_position, 0xff, map_bytes);
    for (uint32_t k = 0; k < count; k++) {
        uint64_t previous = 0;
        for (uint64_t i = 0; i < parts[k].docs; i++) {
            uint64_t position = parts[k].positions[i];
            if (position >= expected_docs || (i && position <= previous)
                    || part_for_position[position] != UINT32_MAX) {
                fprintf(stderr, "invalid/duplicate part position %llu\n",
                        (unsigned long long)position); return 2;
            }
            part_for_position[position] = k; previous = position;
        }
    }
    int out = open(argv[2], O_WRONLY | O_CREAT | O_TRUNC, 0644); if (out < 0) die(argv[2]);
    int lengths = -1;
    if (argc == 6) { lengths = open(argv[5], O_WRONLY | O_CREAT | O_TRUNC, 0644); if (lengths < 0) die(argv[5]); }
    const size_t buffer_size = 64u << 20; unsigned char *buffer = malloc(buffer_size); size_t used = 0;
    uint64_t written_tokens = 0;
    for (uint64_t position = 0; position < expected_docs; position++) {
        uint32_t part_index = part_for_position[position];
        if (part_index == UINT32_MAX) { fprintf(stderr, "missing position %llu\n", (unsigned long long)position); return 2; }
        part_t *p = &parts[part_index]; uint64_t i = p->index;
        if (i >= p->docs || p->positions[i] != position) { fprintf(stderr, "part cursor mismatch\n"); return 2; }
        uint64_t begin = p->offsets[i], end = p->offsets[i + 1];
        if (end < begin || end > p->token_count) { fprintf(stderr, "invalid token offsets\n"); return 2; }
        uint64_t doc_tokens = end - begin;
        if (doc_tokens > UINT32_MAX) { fprintf(stderr, "document exceeds uint32 token length\n"); return 2; }
        if (lengths >= 0) {
            uint32_t value = (uint32_t)doc_tokens;
            if (write(lengths, &value, sizeof(value)) != sizeof(value)) die("write doc length");
        }
        size_t bytes = (size_t)(end - begin) * width; const unsigned char *source = p->tokens + (size_t)begin * width;
        if (bytes > buffer_size) { flush_buffer(out, buffer, &used); size_t done = 0; while (done < bytes) { ssize_t n = write(out, source + done, bytes - done); if (n <= 0) die("write large doc"); done += (size_t)n; } }
        else { if (used + bytes > buffer_size) flush_buffer(out, buffer, &used); memcpy(buffer + used, source, bytes); used += bytes; }
        written_tokens += end - begin; p->index++;
    }
    flush_buffer(out, buffer, &used); fsync(out); close(out);
    if (lengths >= 0) { fsync(lengths); close(lengths); }
    printf("%llu\n", (unsigned long long)written_tokens);
    for (size_t k = 0; k < count; k++) {
        part_t *p = &parts[k];
        if (p->docs) munmap(p->positions, (size_t)p->docs * 8);
        munmap(p->offsets, (size_t)(p->docs + 1) * 8);
        if (p->token_bytes) munmap(p->tokens, p->token_bytes);
        if (p->pfd >= 0) close(p->pfd);
        if (p->ofd >= 0) close(p->ofd);
        if (p->tfd >= 0) close(p->tfd);
    }
    munmap(part_for_position, mapped_bytes); free(buffer); free(parts); return 0;
}
