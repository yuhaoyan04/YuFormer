#!/usr/bin/env bash
# Build the snapshot-bound M3 tokenizer corpus and train the three BBPE E0 arms.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="${REPO_ROOT}/data"
PYTHON_BIN="${PYTHON_BIN:-python}"
CORPUS_OUT="${YUFORMER_TOKENIZER_CORPUS:-${DATA_ROOT}/tokenizer_corpus_m3_20261019}"
TOKENIZER_OUT_ROOT="${YUFORMER_TOKENIZER_CANDIDATES:-${REPO_ROOT}/tokenizer/tokenizers/candidates/m3-20261019}"
SNAPSHOT="${YUFORMER_DATA_SNAPSHOT:-${DATA_ROOT}/processed/releases/mix_m3_e1_100b-20261019.release.json}"
ORDER="${YUFORMER_S6_ORDER:-${DATA_ROOT}/processed/mix_order/mix_m3_e1_100b_stage2.parquet}"
LOG="${DATA_ROOT}/tokenizer_train_m3_20261019.log"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" data-process/scripts/sample_tokenizer_corpus.py \
  --order "${ORDER}" \
  --s3-root "${DATA_ROOT}/processed/s3" \
  --snapshot "${SNAPSHOT}" \
  --recipe tokenizer/configs/corpus_m3_coverage_balanced.yaml \
  --out-root "${CORPUS_OUT}" >>"${LOG}" 2>&1

"${PYTHON_BIN}" data-process/scripts/audit_tokenizer_corpus.py \
  --corpus "${CORPUS_OUT}" \
  --snapshot "${SNAPSHOT}" \
  --order "${ORDER}" \
  --recipe tokenizer/configs/corpus_m3_coverage_balanced.yaml \
  --val-dir "${DATA_ROOT}/processed/val_sets" \
  --out "${DATA_ROOT}/processed/audits/tokenizer_corpus_m3_20261019.json" \
  >>"${LOG}" 2>&1

for cfg in yf-32k-bpe yf-64k-bpe yf-128k-bpe; do
  "${PYTHON_BIN}" tokenizer/scripts/train_tokenizer.py \
    --config "${cfg}" \
    --corpus "${CORPUS_OUT}" \
    --output "${TOKENIZER_OUT_ROOT}/${cfg}" \
    --formal >>"${LOG}" 2>&1
done
