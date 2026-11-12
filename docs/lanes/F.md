# Lane F report (docs and release), 2026-10-31

LANE: F (README, docs, HF cards, self-play). Branch lane-F/docs-release.

TASKS_DONE
1 README.md: Mermaid architecture (gateway, GUARD_URL-agnostic guard, MLflow aliases, Kaggle, Neon, Langfuse, Grafana OTLP, hourly canary), honest Status table (L1-2 live, L3-5 need a guard host, 300 ms NOT met: shipped prefix-KV rung p95 2326 ms via marker, p50 443 ms, 2-vCPU emulation, ADR-0064), headline table with markers for regex, Prompt-Guard, Gemini judge, DeBERTa, Qwen 0.6B/1.7B zero-shot, 0.6B SFT, 1.7B SFT, in-distribution caveat, seeds 0/1/3 + seed 2 divergence, quickstart, links to ADRs/runbook/report/deploy-guard. `scripts/check_readme_numbers.py` passes; `readme_numbers.py fill` run after merging main (scorecard now carries the prefix-KV p95).
2 `scripts/make_hf_cards.py` rewritten: variant (SFT-only, fp32 + prefix-KV), T=1.41 / threshold 0.063, 3-seed mean+-sd, latency not met, ADR-0013 license table (wildjailbreak excluded), gold double-check 92.7% / kappa 0.851, repo layout, intended use, limitations. Every number is parsed from scorecard.json / eval-v1.md / DATASET_CARD.md; a parse miss raises. Pushed README.md only to Pseudo-Harsh/breachlab-guard-v1 and Pseudo-Harsh/breachlab-data (both verified still private).
3 docs/demo-script.md (Script A = works today, Script B = after the guard host) and docs/runbook.md (current state, where things run, "Pointing GUARD_URL at a host" via scripts/set_guard_url.sh, failure modes; Observability section untouched).
4 `scripts/link_check.py`: 0 FAIL, 4 WARN (private GitHub / HF, by design). Relative doc links all exist (docs/deploy-guard.md now present).
5 Self-play on levels 1-2 via breachlab-api.vercel.app, Gemini attacker, no 429/503: 57 sessions (nicknames selfplay-*), 510 attempts, 10 level-1 leaks, 51 regex blocks at L2, 0 L2 clears. Judge (16 Gemini calls, no failures) labeled 124 attempts (117 UNSAFE, 7 SAFE), 66 queued for review; labels written back to Neon over HTTPS. build_dataset on the mirror: 58 collected, 54 after dedup: train 50 (46 UNSAFE), val 4, all source=selfplay; not committed. Detail in docs/adr-drafts/F-selfplay-levels-1-2.md.
6 ruff, ruff format (touched python), mypy strict (95 files) clean; pytest 457 passed / 9 skipped, coverage 98.83%.

TESTS: pass, cov 98.83% (new tests/scripts/test_make_hf_cards.py, 2 tests; the file is outside Lane F dirs, additive only)
BRANCH: lane-F/docs-release (pushed; sha in the final message)

REQUESTS
a) scripts/readme_numbers.py: marker regex `[\w.\-]+` cannot match the scorecard key `llama-prompt-guard-2-86m (Groq)`; change to `MARK = re.compile(r"<!--num:([^<>]+)\.(\w+)-->(.*?)<!--/num-->")`. The README already carries those markers (inert until then; verified equal with the patched regex).
b) docs/blockers.md: "HF visibility | model repo, dataset repo and GitHub repo are private; making them public is irreversible | Harsh flips visibility when ready, after deciding the weights/dataset license (dolly subset is CC-BY-SA-3.0 share-alike, LLM-generated rows follow provider terms), then `LINKCHECK_STRICT=1`."
c) Flywheel (Lane C): add `--attacker gemini` to `breachlab.flywheel.selfplay.main`, and let `judge.py` / `build_dataset.py` mains fall back to Neon HTTPS SQL when TCP is blocked; both were done here with a scratch driver.
d) CLAUDE.md / progress: self-play run 1 and card push done; README Status is the public statement of what is live.

ADR_DRAFTS: docs/adr-drafts/F-cards-from-repo-files.md, F-selfplay-levels-1-2.md

BLOCKED: public release of repo and HF repos (owner action, irreversible); license decision for weights trained on CC-BY-SA data; human review of the 300 + 93 gold rows; levels 3-5 and their leak suite until a guard host exists.
RUNNING_JOBS: none.
NEXT: after the lane A2/B2 final merges rerun `uv run python scripts/readme_numbers.py fill && uv run python scripts/check_readme_numbers.py` and `uv run python scripts/make_hf_cards.py`; once a guard host exists follow the runbook section and update README Status; after the owner flips visibility run link_check with LINKCHECK_STRICT=1.
