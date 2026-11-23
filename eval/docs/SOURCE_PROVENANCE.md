# Evaluation source provenance

The general evaluation backend is EleutherAI `lm-evaluation-harness` v0.4.13 at commit
`ddd67220430a2470529f25fd5c05a576ca1057a0`. The tag and commit are recorded in
`../backend.lock.json`; floating branches are not accepted for publishable runs.

v0.4.13 is the minimum accepted release because its release notes include fixes for evaluation
documents leaking into their own few-shot demonstrations, a multiple-choice filter error, and
group standard-error calculation. Upgrading the harness requires rerunning the frozen suite; scores
from different harness commits must be separate rows rather than silently merged.

YuFormer does not vendor the harness. The adapter must preserve upstream task outputs and adds a
content-addressed result manifest plus raw-sample and contamination gates. Dataset revisions and
task/prompt hashes are locked per run because pinning the harness alone does not pin benchmark data.

Upstream: <https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.13>
