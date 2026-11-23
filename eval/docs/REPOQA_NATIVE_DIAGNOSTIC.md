# Native-tokenizer RepoQA stage diagnostic

This is a stage diagnostic, not a published full RepoQA benchmark score.
It uses the official `evalplus/repoqa` source at
`ae876deb1365dbf5a15b0533723c8ed123eee586` and its versioned
`2024-06-23` release. The downloaded release actually contains six languages.
Both the upstream source and the original compressed and decoded dataset bytes
must be bound by their locks before any case is constructed.

The case protocol is fixed before measuring a model:

- Use all six release languages, with eight eligible functions per language.
  Order candidates by BLAKE2b of an explicit seed, language, repository, and
  function identity. Select the first eight satisfying the same source and
  context checks at 4K, 16K, and 64K. Do not select using a model score.
- Use the unmodified upstream dependency ordering, context construction and
  instruction template. Replace only its hardcoded external tokenizer factory
  with the actual immutable YuFormer tokenizer. Import selected source AST
  definitions without executing upstream model clients or download code.
- Require the upstream character and byte slices to agree with the line slice
  after removing only the first line's leading spaces/tabs. Tree-sitter ranges
  start at the function keyword, while line ranges include its indentation.
  Preserve all remaining whitespace and line endings. The original builder slices Python strings
  using fields named `start_byte` and `end_byte`; reject incompatible records
  rather than silently changing upstream context construction.
- Reserve 768 generated tokens. The reference function may occupy at most
  512 native tokens. Admit only contexts containing the complete reference,
  with a true whole-prompt length of at least 75% of the requested context
  capacity, and with prompt plus reservation within that capacity.
  The upstream sum of per-line token counts overestimates native whole-context
  token counts. Try the fixed budget multipliers 1.00, 1.03, 1.06, 1.09, 1.12
  in that order, starting at capacity minus generation, instruction/description
  tokens and 64 tokens of formatting space. Select the first context satisfying
  both unchanged native length constraints; otherwise reject the function.
- Pair each context case with the exact same description and instruction but
  no repository context. Use identical cases and deterministic greedy decoding
  for actual baseline and candidate checkpoints.
- Preserve the upstream function-matching scorer: fenced-function extraction
  with Tree-sitter, whitespace BLEU with NLTK smoothing method 4, best target
  among all annotated functions in that repository, and similarity >= 0.8.
  Record exact string match as a separate diagnostic if desired; do not call
  another metric the official function-matching score.

The no-context result is a control, not a contamination certificate. The
existing training decontamination release does not certify removal of RepoQA.
Do not claim that these public repositories were absent from pretraining.
Report context and no-context results, their paired changes, per-language
coverage, and the limited sample count together.

The training source, schedule, mixture and stage qualification are unchanged.
These assets do not grant a capability admission. Measurements, independent
scoring and a separately frozen quantitative stage policy remain required.
