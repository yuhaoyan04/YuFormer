# Invalidated E0-Q2 configs

These compatibility-backend runs are retained only for checkpoint migration and throughput forensics.
Their machine-readable invalidation record is
`data/experiments/e0q2/INVALIDATED_20261018.json`: validation used the wrong tokenizer, one arm leaked
training data into validation, token streams lacked document EOD, output directories had concurrent
writers, and the tokenizer/chat contract has since changed. Do not resume them or use their metrics for
tokenizer/model selection.
