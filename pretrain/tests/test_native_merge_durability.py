import os
from pathlib import Path
import struct
import subprocess

import pytest


@pytest.fixture(scope="module")
def merger(tmp_path_factory):
    directory = tmp_path_factory.mktemp("native-merge-durability")
    production = Path(__file__).resolve().parents[1] / "scripts/merge_token_parts.c"
    source = Path(os.environ.get("YUFORMER_TEST_NATIVE_MERGE_SOURCE", production))
    wrapper = directory / "faults.c"
    wrapper.write_text(r'''
#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
extern int __real_fsync(int fd);
extern int __real_close(int fd);
static int fails(int fd, const char *operation) {
    char name[64], target[4096], mode[64];
    const char *requested = getenv("YUFORMER_TEST_WRITEBACK_FAILURE");
    if (!requested) return 0;
    snprintf(name, sizeof(name), "/proc/self/fd/%d", fd);
    ssize_t n = readlink(name, target, sizeof(target)-1);
    if (n < 0) return 0;
    target[n] = 0;
    const char *kind = strstr(target, "output.tokens") ? "tokens" :
                       strstr(target, "output.lengths") ? "lengths" : NULL;
    if (!kind) return 0;
    snprintf(mode, sizeof(mode), "%s_%s", operation, kind);
    if (strcmp(requested, mode)) return 0;
    errno = EIO;
    return 1;
}
int __wrap_fsync(int fd) { return fails(fd, "fsync") ? -1 : __real_fsync(fd); }
int __wrap_close(int fd) { return fails(fd, "close") ? -1 : __real_close(fd); }
''')
    binary = directory / "merge"
    subprocess.run(["cc", "-O2", "-std=c11", str(source), str(wrapper),
                    "-Wl,--wrap=fsync", "-Wl,--wrap=close", "-o", str(binary)], check=True)
    return binary


def invoke(merger, directory, failure=None):
    manifest, lines = directory / "manifest.tsv", []
    for number, (positions, offsets, tokens) in enumerate([
        ([0, 2], [0, 2, 4], [11, 0, 33, 0]), ([1], [0, 2], [22, 0])
    ]):
        paths = [directory / f"part{number}.{kind}" for kind in ("positions", "offsets", "tokens")]
        for path, values, kind in zip(paths, (positions, offsets, tokens), ("Q", "Q", "H")):
            path.write_bytes(struct.pack("<" + kind * len(values), *values))
        lines.append("\t".join(map(str, paths)) + f"\t{len(positions)}\t{len(tokens)}")
    manifest.write_text("\n".join(lines) + "\n")
    env = dict(os.environ)
    env.pop("YUFORMER_TEST_WRITEBACK_FAILURE", None)
    if failure:
        env["YUFORMER_TEST_WRITEBACK_FAILURE"] = failure
    return subprocess.run([str(merger), str(manifest), str(directory / "output.tokens"),
                           "3", "2", str(directory / "output.lengths")],
                          capture_output=True, text=True, env=env, timeout=10)


def test_checked_native_merge_preserves_exact_document_order_and_ledger(merger, tmp_path):
    result = invoke(merger, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "6"
    assert (tmp_path / "output.tokens").read_bytes() == struct.pack("<6H", 11, 0, 22, 0, 33, 0)
    assert (tmp_path / "output.lengths").read_bytes() == struct.pack("<3I", 2, 2, 2)


@pytest.mark.parametrize("failure", ["fsync_tokens", "fsync_lengths", "close_tokens", "close_lengths"])
def test_writeback_or_close_error_prevents_success_receipt(merger, tmp_path, failure):
    result = invoke(merger, tmp_path, failure)
    assert result.returncode != 0
    assert not result.stdout.strip()
    assert "Input/output error" in result.stderr
