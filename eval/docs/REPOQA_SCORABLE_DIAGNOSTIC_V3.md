# Scorable native RepoQA diagnostic, version 3

This version retains the native diagnostic protocol in
`REPOQA_NATIVE_DIAGNOSTIC.md`, including its seed ordering, six languages,
eight functions per language, three contexts, 75% floor, 768 generated tokens,
512-token reference cap, and paired no-context controls.

Add one source-level eligibility condition before selecting any function:
the original upstream scorer must identify that function when given its exact
original reference wrapped in its language's code fence, and must give
similarity >= 0.8. This check uses reference bytes only, without model output.
Record every rejected candidate and preserve the fixed BLAKE2b ordering.

The prior version 2 source/context audit passed, but a subsequent independent
reference self-consistency check found one C++ constructor that the upstream
Tree-sitter function extractor misidentifies. Its gold output received BLEU
0.5757197301274735. Preserve that manifest and audit as historical case-asset
evidence; do not use that set for stage admission. Keep the original scorer
and threshold unchanged. Version 3 is a separately materialized diagnostic
subset, with explicit parser/scorer dependency hashes.

Passing this source check is not a model result and provides no contamination
guarantee. Actual candidate/baseline measurements and capability admission
remain future work after the real checkpoints exist.
