# Guard v2 evaluation (lane T, ADR-0094)

v2 = the same recipe as v1 (Qwen3-0.6B, LoRA r16, SFT only, seed 0, 3 epochs, same hyper-parameters) trained on data-v2: v1 plus 2,172 new benign rows (short hard-benign openers in en/hi/hinglish, longer hi/hinglish benign text). Served as the int4 q4b32 graph, recalibrated on the v2 validation split (temperature 1.124, threshold 0.96458, `threshold_for_fpr(0.02)` on val; never tuned on gold_test or the opener set).

Sources: `docs/lanes/T-guard-v2-gates.json` (numbers), `scripts/v2_gates.py` (script), Kaggle kernels `breachlab-sft-06b-v2-s0`, `breachlab-export5-v2kv`, `v2q-val`, `v2q-gold`, `v2q-open` (job IDs in `docs/progress.md`).

## v1 vs v2 (served int4 graph)

| Metric | v1 (champion) | v2 | Gate |
|---|---|---|---|
| gold_test ASR (n=1500, 682 attacks) | 0.59% | **0.44%** [0.0, 1.03] | <= v1 + 0.5pp (1.09%): PASS |
| gold_test FPR | 2.08% | **1.83%** [0.98, 2.81] | <= 2.5%: PASS |
| gold_test AUROC | 0.999 | 0.9998 | n/a |
| Held-out short benign openers FPR (n=300) | 54.3% [48.7, 59.7] | **6.3%** [4.0, 9.3] | <= 10%: PASS |
| of which Hindi (n=100) | 96% | **15.0%** [8, 22] | <= 15%: PASS (at the bar, see caveat) |
| of which Hinglish (n=100) | 48% | 2% [0, 5] | n/a |
| of which English (n=100) | 19% | 2% [0, 5] | n/a |
| Validation FPR / ASR at the threshold | 2.0% / n/a | 1.99% / 0.86% | n/a |

Opener FPR by category (v2): greeting 10.7%, meta question 8.9%, small talk 2.7%, simple request 1.7% (v1: 45-69%).

## Caveats (read these)
- The Hindi opener FPR sits exactly at the 15% bar; its interval is 8-22%. The remaining Hindi blocks are mostly greetings ("नमस्ते! आपका दिन कैसे शुरू हुआ?", "हे वहाँ! क्या चल रहा है?"). The next rung (more Hindi greetings, a second epoch of the new rows) is not taken because the gate passes; it is the obvious v3 data target.
- The opener set was held out of every training and validation file (dedup by MinHash during the v2 build), but it is only 300 LLM-generated rows; it is a regression detector, not an out-of-distribution proof.
- gold_test is unchanged from v1 and still shares sources with training (same caveat as eval-v1); the out-of-distribution set in the cascade section above (n=934, ADR-0106) is the check that does not. The 0.44% vs 0.59% ASR difference is inside the intervals; v2 is "no worse", not "better".
- The data build ran at half the planned volume (2,172 of about 4,000 new rows) to stay inside the free-tier model quotas.
- One seed only; v1 reported three. No int4 parity run against fp32 for v2 (`--no-reference`): only the served int4 graph was scored.
- The temperature is lower (1.124 vs 1.412) and the attack scores are far more saturated (median attack p_unsafe 0.99999 vs 0.9988), so thresholds tuned on v1 scores (the level 3 relaxed threshold, ADR-0093) had to be re-derived.

## Student vs 0.6B vs cascade (lane D, ADR-0105)

The 0.6B guard (teacher) is distilled into a 136 MB multilingual DistilBERT int8 ONNX student (KD on teacher soft labels, T=3, plus hard labels, attack weighting, game-attack mining pool). The guard function serves a cascade: the student scores first (calibrated temperature 0.905); score below t_low=0.1337 returns SAFE, above t_high=0.8939 returns UNSAFE, the uncertain band between them escalates to the 0.6B (fail-closed UNSAFE if it is unavailable). Output checks always use the 0.6B. Thresholds were chosen on val plus 130 separate tuning attacks under a 5% cap on escalation for half-benign, half-attack traffic; tuning attacks, suite (216) and fresh (110) game attacks, opener slice and OOD set were not used to pick them. Intervals are 95% bootstrap. Numbers use the same served int4 teacher scores on every set (`docs/lanes/D-cascade-v2.json`, `scripts/cascade_report.py`).

| metric | 0.6B teacher | distilbert student | cascade |
|---|---|---|---|
| gold_test ASR (n=1500) | 0.4% [0.0, 0.9] | 0.3% [0.0, 0.7] | 0.4% [0.0, 0.9] |
| gold_test FPR | 1.8% [1.0, 2.8] | 1.8% [1.0, 2.8] | 1.1% [0.4, 1.8] |
| gold_test AUROC | 0.9998 | 0.9997 | 0.9996 |
| val ASR | 0.9% [0.4, 1.4] | 0.9% [0.4, 1.5] | 0.8% [0.3, 1.3] |
| val FPR | 2.0% [1.3, 2.8] | 2.0% [1.3, 2.7] | 1.6% [1.0, 2.3] |
| opener FPR (n=300) | 6.3% [4.0, 9.3] | 10.0% [6.7, 13.7] | 6.3% [4.0, 9.3] |
| game attacks, tuning split (n=130) ASR | 1.5% [0.0, 3.8] | 6.9% [3.1, 11.5] | 3.1% [0.8, 6.2] |
| game attacks, template suite (n=216) ASR | 1.9% [0.5, 3.7] | 6.0% [2.8, 9.3] | 5.6% [2.8, 8.8] |
| game attacks, fresh (n=110) ASR | 6.4% [2.7, 11.8] | 21.8% [14.5, 29.1] | 14.5% [8.2, 20.9] |
| OOD ASR (n=934) | 6.6% [4.5, 9.1] | 13.4% [10.5, 16.3] | 9.5% [7.0, 12.0] |
| OOD FPR | 17.3% [14.0, 20.7] | 20.9% [17.3, 24.9] | 17.8% [14.4, 21.1] |
| OOD AUROC | 0.9593 | 0.9105 | 0.9169 |
| Hindi opener FPR (n=100) | 15.0% | 21.0% | 15.0% |

Escalation to the 0.6B by set: val 4.5%, gold 2.9%, openers 23.0%, tuning attacks 14.6%, template suite 9.3%, fresh attacks 27.3%, OOD 22.4%. Production mix (100 real requests, half attacks): 5 of 104 guard calls (4.8%).

Model size and time (Kaggle 2 vCPU, ONNX Runtime, one thread; production p50/p95 below): student 136 MB, p50 18-43 ms, p95 27-144 ms per message; 0.6B int4 433 MB, p50 about 720 ms, p95 about 2,063 ms on the 1 vCPU Vercel host.

Production, region sin1 (next to Neon), 100 requests through the full gateway at level 4, concurrency 1, 50% attacks (`docs/lanes/D-full-request-cascade.json`): client round trip from this sandbox p50 2,721 ms, p95 4,027 ms; the gateway's own server time p50 527 ms, p95 1,142 ms; 0 of 49 benign blocked, 46 of 49 attacks blocked at the guard; 2 of 100 answered 503 (vault busy). Before (v1 guard, iad1, shadow call): p50 5,053 ms, p95 6,447 ms.

What this says, without rounding:
- On the gold set (in-distribution) the cascade equals the 0.6B in quality and escalates 3% of the time.
- Out of distribution and on the fresh game attacks the student is clearly weaker (OOD ASR 13.4% vs 6.6%; fresh 21.8% vs 6.4%) and the cascade closes about half of the gap (9.5%, 14.5%). The student's confident calls are the problem, not the band. Quality on novel attacks is the price of the speed.
- The 300 ms guard p95 holds only while escalation stays under 5% (mixed traffic: student p95 is under 150 ms). On opener-like, fresh-attack or OOD traffic 9-27% of requests reach the 0.6B and the p95 is the 0.6B's (about 2 s). It is recorded as DONE-BY-DECISION, not met.
- Not measured, and why: the 500-request guard-only benchmark at concurrency 1 and 4 needs the guard's bearer token, which this session does not hold (`scripts/bench_vercel_guard.py` is ready); the full-request run was 100 requests at c1 to protect the shared Groq quota (200k tokens a day). Cold start: the student downloads in 11.7 s and the 0.6B in 38 s on first use after an idle period.

## Rollback
v1 stays at `champion/` in `Pseudo-Harsh/breachlab-guard-v1`; v2 is in `champion-v2/`. The guard host selects the folder with the `GUARD_ALIAS` env var (`champion` = v1, `champion-v2` = v2) and redeploys; no code change.
