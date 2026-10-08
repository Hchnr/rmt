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

## Reproduction

Use the existing pinned environments (`.venv-cached` for training/inference; `.venv-eval` for official scoring). Set `RMT_ALLOWED_GPUS=4,5,6,7`; launch GPU work through `scripts/run_06b_stage.py` with a new stage name and an explicit free authorized device. Never overwrite completed stage names or run another training job on devices occupied by evaluation replicas.

```sh
.venv-cached/bin/python scripts/audit_training_exposure.py --configs configs/v0.0.5/pilot_ce.yaml configs/v0.0.5/pilot_kl01.yaml configs/v0.0.5/pilot_kl05.yaml configs/v0.0.5/short_kl05.yaml configs/v0.0.5/mixed_kl05.yaml configs/v0.0.5/mixed_kl05_lr1e6.yaml --world-size 1 --output reports/v0.0.5/all_training_exposure.json
CUDA_VISIBLE_DEVICES='' .venv-cached/bin/python scripts/audit_06b_labels.py
.venv-cached/bin/python scripts/summarize_06b_campaign.py
```

After both complete formal run manifests exist:

```sh
.venv-cached/bin/python scripts/summarize_06b_formal.py --native reports/v0.0.5/full_native06b_v2_runs.json --candidate reports/v0.0.5/full_mixed_lr1e6_final_runs.json --output reports/v0.0.5/formal_comparison.json
```

This rejects incomplete benchmarks, duplicate/missing scores, different request identities, different generation semantics, and incompatible inference environments. Exported generation files differ in formatting/version metadata, so each is validated against its own recorded hash and then compared semantically.

## Acceptance status

| Area | Status | Boundary |
| --- | --- | --- |
| Pinned0.6B conversion and parameter accounting | Passed | Logical ties audited separately from duplicate native checkpoint storage |
| Eager reference, cache, compile replay, packed training regressions | Passed on tested inputs | No universal cross-backend/shape bitwise guarantee |
| Three matched1M loss recipes and follow-up attribution | Executed | No candidate passes both predeclared quality gates |
| Data/label/exposure audits | Passed | Available corpus size differs from actually visited IDs |
| Independent resume/export | All six passed | Trained BF16 export uses CE tolerance, not bitwise identity |
| Full MATH-500 / IFEval | Native/trained complete; untrained RMT running | MATH49.2%→47.0%; IFEval57.67%→49.54%, paired95% CI [−11.83,−4.62]pp; no gain |
|10M/30M fixed32/dynamic main experiment | Not executed, gated | No extra-depth or adaptive-compute benefit established |
| New32B teacher generation | Not executed, gated | Existing accepted answers reused with provenance |

Parameter details: `parameter_audit.json`; supervision examples: `label_audit.json`; resume/export checks: `training_export_resume.json`. Native generation is a Transformers eager control and RMT uses compiled projections, so full-service timing is not an architecture-only speed comparison.

P1 is an original-depth SFT/KL control represented by bound experts:28 recurrences, one original expert per step, no learned routing and no extra-depth training. Its quality changes cannot be attributed to adaptive recurrence or used to reject that architecture. The experiment isolates training behavior before introducing that additional factor.

## SDPA attribution follow-up

`native_compiled_reference.json` and `native_eager_sdpa_diagnostic.json` preserve failed cross-path tolerance checks on unpadded batch1. `native_compiled_reference_aligned.json` uses the same explicit mask and passes all96 comparisons bitwise. This separates mask/kernel behavior from compilation; it does not retroactively pass the original check. A full untrained RMT control is queued with unchanged production source and evaluation protocol.

`fsdp06b_verification.json`: real0.6B two-GPU16-step packed/compiled/checkpointed KL training, exact independent resume, and HF loss-tolerance checks pass. This is engineering validation, not a seventh quality candidate or a matched-global-batch scaling experiment.
