LANE A (eval and data quality) | 2026-10-30
TASKS_DONE
1 seeds: s2 DIVERGED (fp16 NaN, evalgold logits 100% NaN, kept in data/eval/gold_s2); re-run as seed 3 (sft-06b-s3 + evalgold-06b-s3). SFT mean+-sd over seeds 0,1,3: ASR 0.49+-0.17%, FPR 1.83+-0.24%, F1 0.9867+-0.0017, AUROC 0.9995+-0.0002; DPO 0.9872+-0.0008 (reports/eval-v1.md).
2 export2 fetched; parity (300 gold rows): fp32 and weight-only int8 OK, dynamic int8 FAILS. Winner = ONNX fp32 of the SFT adapter (docs/lanes/A-variant-spec.md). export2 was the DPO adapter: SHIPPING export must be sft-0.6b-s0 (ADR-0036), Lane B re-exports it.
3 1.7B measured with my own CPU kernel (giriharsh/breachlab-exporta-sft17, SFT 0.6B + 1.7B on one machine): "0.6B keeps 99.6% of 1.7B's F1 at 2.5x lower p95 (2.7x lower p50)": fp32 p50/p95 733/3112 ms vs 1998/7769 ms. 300 ms gate not met by any parity-safe variant.
4 gemini-judge row 12.8% ASR, 2.8% FPR, F1 91.5% (1500/1500). Prompt-Guard row done. Leak suite: L1 n=210 1 leak (0.5%), L2 n=216 0 leaks (agent gpt-oss-20b, temp 0); L3-L5 not run (no GUARD_URL, no Space exists). Llama Guard dropped (ADR draft).
5 gold double-check: agreement 92.7%, kappa 0.851, 110 disagreements (87 gen-UNSAFE judged SAFE, 23 gen-SAFE judged UNSAFE); 93 appended to Neon gold-review-v1, 17 were already in the 300 sample; cache data/eval/gold_judge.json.
6 reports/eval-v1.md, web/public/scorecard.json, reports/gate-input.json regenerated from real files; dvc add/push data/eval (md5 07e76008..., 55 files, pushed).
TESTS pass 139, coverage 97.5%, ruff + mypy strict clean (new: eval/parity.py, eval/gold_judge.py, tests/eval/test_{parity_judge,export3,lane_a_scripts}.py)
BRANCH lane-A/eval-quality (sha in git log; pushed)
REQUESTS (orchestrator/Lane B, exact diffs)
 a) src/breachlab/train/kernel_main.py: add `ap.add_argument("--which", default="dpo-0.6b-s0")` and pass `args.which` as 4th arg to run_export. src/breachlab/train/pipeline.py run_export: `sample = gold[:300]` -> `sample = gold` (parity on all 1,500 rows).
 b) Training stability: configs/train.yaml sft lr 2e-4 -> 1e-4 (fp32 LoRA is already used; seed 2 NaN at epoch 0.66, seed 3 skipped 3 NaN steps); T4 has no usable bf16.
 c) Progress/CLAUDE.md: Phase 3 seeds are {0,1,3}; export winner = fp32 SFT; adopt ADR drafts A-*.md. Kaggle slugs lane-prefixed (A-kaggle-slug-namespacing.md): Lane B and I both pushed breachlab-export3-sft and one overwrote the other.
ADR_DRAFTS docs/adr-drafts/A-{seed2-divergence,llama-guard-dropped,gold-judge,kaggle-slug-namespacing,export-variant-fp32}.md
BLOCKED L3-L5 leak suite: needs a live GUARD_URL (Lane B's Space; none under Pseudo-Harsh yet); run `uv run python scripts/run_leak_suite.py` with GUARD_URL set (resumes, only L3-L5 left). Human review of the 300+93 queued rows is Harsh's.
RUNNING_JOBS none (all Kaggle kernels COMPLETE: sft-06b-s2/s3, evalgold-06b-s2/s3, exporta-sft17, export2-06b-s0)
NEXT after Lane B delivers GUARD_URL: L3-L5 suite; re-run scripts/make_report.py; if the human review changes gold labels, re-run finalize_data/make_report; full-1500 parity once REQUEST a lands.
