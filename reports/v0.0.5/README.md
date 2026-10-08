# v0.0.5 review guide

Student/base/online KL teacher: pinned Qwen3-0.6B. All jobs use physical GPUs 4–7. Parameter count: 596,049,920 base + 29,680 router = 596,079,600 unique parameters.

## Executed results

- `reference.json`: tested native/fixed28 eager logits, cached decode and HF reload are bitwise identical.
- `dynamic_reference.json`: tested forced token depths28–36, cache/full and compile/eager correctness.
- `deterministic_compiled_decode.json`, `deterministic_eager_repeat.json`: deterministic replay has zero measured error. Earlier `compiled_decode_*` and `eager_decode_repeat.json` are diagnostic failures under the former nondeterministic configuration, not final-path acceptance.
- `native_cache_verification.json`: preallocated native cache versus HF DynamicCache, 96 real-model comparisons, bitwise equality.
- `tokenizer_corpus_audit.json`: all23,104 records match official tokenizer IDs and decoded text.
- `all_training_exposure.json`: independently reconstructed pack membership agrees with actual training totals.
- `campaign_summary.json`: paired development scores, conditional bootstrap intervals, quality gate and stage reservation accounting. Regenerate after running stages finish with `scripts/summarize_06b_campaign.py`.
- `deterministic_performance*.json`: final deterministic benchmark; prior `reference_performance*.json` are superseded historical measurements.

## Interpretation

Six independent fixed28 recipes start from original mapped weights. CE/KL0.1/KL0.5 each train1,000,162 supervised tokens over788 distinct IDs. The250k check sees312 IDs. Broad-data runs each train1,000,578 targets over2,600 distinct IDs, not all20,066 available records.

The best development candidate is broad data, KL0.5, LR1e-6: instruction45/48 versus native46/48, math48/48 for both. The2.083pp instruction drop narrowly fails the predeclared2pp engineering gate. This is not evidence of statistically significant harm. Lower dev CE alone is not a quality win.

P2 fixed32/dynamic large-scale training and new32B generation were not executed because no recipe passed the gate. Forced-depth correctness and untrained path diagnostics do not establish learned dynamic-depth benefit. Current evidence proves neither architectural gains nor full quality recovery.

Frozen full MATH-500/IFEval runs are tracked separately. Interrupted `full_native06b_*` contains only partial historical results and must not be reported as a completed baseline or combined with the new engine's responses. Formal scores are diagnostic only and do not reopen development selection.

Raw training/probe reports preserve full evidence. Stage costs include allocated device lifetime, not only kernel time; nested wrapper cost reports must not be added again. Existing v0.0.4 teacher generation costs are historical and excluded from new-stage totals.
