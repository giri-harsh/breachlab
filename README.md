# BreachLab

[![CI](https://img.shields.io/github/actions/workflow/status/giri-harsh/breachlab/ci.yml?branch=main&label=ci)](https://github.com/giri-harsh/breachlab/blob/main/.github/workflows/ci.yml)
[![Code: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-blue.svg)](LICENSE)
[![Data: CC-BY-SA-3.0](https://img.shields.io/badge/data-CC--BY--SA--3.0-lightgrey.svg)](DATA_LICENSE.md)
[![Play](https://img.shields.io/badge/play-live%20demo-ff2e93.svg)](https://breachlab-web.vercel.app)
[![HF model](https://img.shields.io/badge/%F0%9F%A4%97-model-yellow.svg)](https://huggingface.co/Pseudo-Harsh/breachlab-guard-v1)
[![HF dataset](https://img.shields.io/badge/%F0%9F%A4%97-dataset-yellow.svg)](https://huggingface.co/datasets/Pseudo-Harsh/breachlab-data)

<p align="center">
  <a href="https://breachlab-web.vercel.app"><img src="docs/media/demo.gif" alt="BreachLab demo: home, boss select, a level 1 leak, skipping a boss" width="640"></a><br>
  <sub>15 s recorded on production: home, boss select, a level 1 leak, skip boss (<a href="docs/media/demo.mp4">MP4</a>)</sub>
</p>

**BreachLab** is a web game where you try to trick a guarded AI vault into leaking a secret code across five bosses of rising defence. The core is a small fine-tuned **guard model** (Qwen3-0.6B + LoRA, int4 ONNX, CPU) that classifies every message as safe, injection, jailbreak or exfiltration, plus the MLOps around it: data versioning, a measured eval gate, a model registry with rollback, and a canary. It runs on free tiers ($0).

> **Research demo.** Not a claim of production safety. Prompts are logged and used to train the guard; do not enter personal data.

## Quick start
- **Play:** https://breachlab-web.vercel.app (gateway `/api/health`: https://breachlab-api.vercel.app/api/health)
- **Run it locally:** [`docs/LOCAL_DEV.md`](docs/LOCAL_DEV.md)
- **Under the hood:** the site's *How it works* and *Model card* pages, [`reports/eval-v2.md`](reports/eval-v2.md), [`docs/decisions.md`](docs/decisions.md)

<p align="center"><img src="docs/media/screens-grid.png" alt="Six production screens: home and boss select, play, boss defeated, skip boss, high scores, under the hood" width="900"><br>
<sub>Neon Arcade UI at 1440 px: home, play, boss defeated, skip, high scores, under the hood (captured from the production build; some screens use sample data from the mocked API).</sub></p>

## How it works
- **Five bosses plus a daily boss.** 1: system prompt only. 2: a keyword wall. 3: the guard with a relaxed threshold. 4: the guard at its calibrated threshold. 5: the guard plus an output filter. A free-tier open LLM on Groq plays the vault and holds a per-session secret.
- **One forward pass guard.** Qwen3-0.6B + LoRA (SFT) scores a message with a softmax over four verbalizer tokens; the served graph is int4 ONNX with prefix-KV reuse on a 1 vCPU Vercel Python function.
- **Free labels.** A deterministic leak detector (base64, hex, ROT13, leetspeak, reversed, separators) decides every win, so each attempt becomes a correctly labelled training row (with a consent line).
- **Measured releases.** Training runs on Kaggle T4; an eval gate checks ASR, FPR, ECE and latency; MLflow aliases (champion, challenger, previous) move the served model, rollback measured at a median 4.9 s (`docs/lanes/P-rollback-timing.json`).
- **Fails closed and says so.** If the guard is down, levels 3-5 refuse rather than let attacks through; a cold guard shows "BOSS WAKING UP" (about 22 s measured on first load), not an error. Skip boss, hints, points and a leaderboard keep it playable.

## Results (gold_test, 1,500 rows; v1 from `reports/eval-v1.md`, v2 from `reports/eval-v2.md`)

| model | attack success rate (lower is better) | false-positive rate | F1 | AUROC |
|---|---|---|---|---|
| regex filter | <!--num:regex.asr-->59.4%<!--/num--> | <!--num:regex.fpr-->10.3%<!--/num--> | <!--num:regex.f1-->0.531<!--/num--> | <!--num:regex.auroc-->0.652<!--/num--> |
| Llama Prompt-Guard-2-86M (via Groq) | <!--num:llama-prompt-guard-2-86m (Groq).asr-->51.6%<!--/num--> | <!--num:llama-prompt-guard-2-86m (Groq).fpr-->10.3%<!--/num--> | <!--num:llama-prompt-guard-2-86m (Groq).f1-->0.602<!--/num--> | <!--num:llama-prompt-guard-2-86m (Groq).auroc-->0.868<!--/num--> |
| Gemini as judge (zero-shot) | <!--num:gemini-judge.asr-->12.8%<!--/num--> | <!--num:gemini-judge.fpr-->2.8%<!--/num--> | <!--num:gemini-judge.f1-->0.915<!--/num--> | <!--num:gemini-judge.auroc-->0.969<!--/num--> |
| DeBERTa prompt-injection classifier | <!--num:deberta-prompt-injection.asr-->10.4%<!--/num--> | <!--num:deberta-prompt-injection.fpr-->14.2%<!--/num--> | <!--num:deberta-prompt-injection.f1-->0.867<!--/num--> | <!--num:deberta-prompt-injection.auroc-->0.904<!--/num--> |
| Qwen3-0.6B zero-shot | <!--num:qwen3-0.6b-zeroshot.asr-->100.0%<!--/num--> | <!--num:qwen3-0.6b-zeroshot.fpr-->0.4%<!--/num--> | <!--num:qwen3-0.6b-zeroshot.f1-->0.000<!--/num--> | <!--num:qwen3-0.6b-zeroshot.auroc-->0.591<!--/num--> |
| Qwen3-1.7B zero-shot | <!--num:qwen3-1.7b-zeroshot.asr-->61.9%<!--/num--> | <!--num:qwen3-1.7b-zeroshot.fpr-->13.2%<!--/num--> | <!--num:qwen3-1.7b-zeroshot.f1-->0.495<!--/num--> | <!--num:qwen3-1.7b-zeroshot.auroc-->0.777<!--/num--> |
| Qwen3-0.6B fine-tuned, fp32 adapter (reference) | <!--num:qwen3-0.6b-sft.asr-->0.6%<!--/num--> | <!--num:qwen3-0.6b-sft.fpr-->1.6%<!--/num--> | <!--num:qwen3-0.6b-sft.f1-->0.988<!--/num--> | <!--num:qwen3-0.6b-sft.auroc-->0.999<!--/num--> |
| Qwen3-0.6B fine-tuned, int4 graph as served (v1, previous) | <!--num:qwen3-0.6b-sft-int4.asr-->0.6%<!--/num--> | <!--num:qwen3-0.6b-sft-int4.fpr-->2.1%<!--/num--> | <!--num:qwen3-0.6b-sft-int4.f1-->0.985<!--/num--> | <!--num:qwen3-0.6b-sft-int4.auroc-->0.999<!--/num--> |
| **Qwen3-0.6B fine-tuned, int4 graph as served (v2, ships)** | <!--num:qwen3-0.6b-sft-int4-v2.asr-->0.4%<!--/num--> | <!--num:qwen3-0.6b-sft-int4-v2.fpr-->1.8%<!--/num--> | <!--num:qwen3-0.6b-sft-int4-v2.f1-->0.987<!--/num--> | <!--num:qwen3-0.6b-sft-int4-v2.auroc-->1.000<!--/num--> |
| Qwen3-1.7B fine-tuned (comparison, not served) | <!--num:qwen3-1.7b-sft.asr-->0.0%<!--/num--> | <!--num:qwen3-1.7b-sft.fpr-->1.3%<!--/num--> | <!--num:qwen3-1.7b-sft.f1-->0.992<!--/num--> | <!--num:qwen3-1.7b-sft.auroc-->1.000<!--/num--> |

Guard v2 (served) vs v1 (`reports/eval-v2.md`): short benign opener false positives **54.3% to 6.3%** (300 held-out rows; Hindi 96% to 15.0%), gold ASR 0.59% to 0.44%, gold FPR 2.08% to 1.83%.

Bootstrap confidence intervals, McNemar tests, slices, the worst misses, the 3-seed table and the size-vs-quality table are in [`reports/eval-v1.md`](reports/eval-v1.md); `scripts/check_readme_numbers.py` checks the table against `web/public/scorecard.json`.

**Game difficulty** (production, guard v2, 110 fresh attacks per boss; template runs in `docs/lanes/G-difficulty-prod.json`, ADR-0102): boss 1 leaks 79.1%, boss 2 57.3%, boss 3 30.0% (21.6% pooled with the 216 template attacks), bosses 4 and 5 0 of 326. Targets: 70+, 40-55, 20-35, 10 or less, 5 or less percent.

### Cost

**Free tiers, $0 today** (Groq free tier, Vercel Hobby, Neon free, Hugging Face free; Gemini is only the offline judge, not in the serving path). Measured from real vault-agent token usage and the guard's measured compute (`scripts/p_cost.py`, `docs/lanes/C-cost.md`), at list prices if usage went past the free tiers: about **$0.11 per 1,000 requests** and **$3.4 per 1,000 games** (a game is five bosses, about 30 messages; $1.7 for a quick game, $9.4 if every boss uses its full 20 messages). Per 1,000 games: guard on Vercel $2.24, vault LLM $0.47 (gpt-oss-20b; $5.1 if qwen3.8-27b serves), gateway $0.60, Neon $0.06. The limit that binds first is the Vercel Hobby Active CPU allowance (4 h a month, about 260 games a month at the measured 1.1 s of guard CPU per call), then Groq's 200k tokens a day per model. A guard with 5-10x lower compute (a cascade) would cost about $1.0 to $1.4 per 1,000 games and lift the Active CPU ceiling to roughly 1,100-1,900 games a month.

## Architecture

```mermaid
flowchart LR
  P[Player phone or laptop] --> W[Next.js on Vercel]
  W --> G[FastAPI gateway on Vercel]
  G -->|level 2| RX[Keyword wall]
  G -->|levels 3-5, POST /score| IG[Guard v2: Qwen3-0.6B LoRA<br/>int4 ONNX, CPU, Vercel function]
  G -->|passed| A[Vault agent: Groq model chain]
  A --> OF[Output filter, level 5, same guard]
  OF --> LD[Leak detector]
  LD --> DB[(Neon Postgres)]
  DB --> R[Judge labels + review] --> T[Train on Kaggle T4: SFT, calibration, int4 export]
  T --> GATE[Eval gate: ASR, FPR, ECE, latency]
  GATE --> MR[MLflow registry on DagsHub<br/>champion, challenger, previous]
  T --> HF[Hugging Face Hub<br/>model and dataset]
  HF -->|weights at boot| IG
  MR -->|alias re-read every 30 s| G
  DVC[DVC data on DagsHub] --> T
  CAN[Canary workflow] -.->|health + probes| G
  G -. optional .-> OBS[Langfuse + Grafana via OTLP]
```

`GUARD_URL` is host-agnostic: any host that serves `POST /score` and `GET /health` ([`docs/CONTRACTS.md`](docs/CONTRACTS.md), [`docs/deploy-guard.md`](docs/deploy-guard.md)) can replace the Vercel function. Tracking and registry: MLflow on DagsHub; data: DVC on DagsHub; training: Kaggle; operations: [`docs/runbook.md`](docs/runbook.md).

## Limitations & honesty
- **Latency misses the bar, but is much better.** The PRD asked for guard p95 <= 300 ms on CPU and full request p95 <= 2 s. The guard is now a cascade: a 136 MB distilled student answers about 95% of ordinary traffic in tens of milliseconds and sends the uncertain rest to the 0.6B (p50 about 720 ms, p95 about 2,063 ms on the free 1 vCPU host). Production, 100 full requests at level 4 in Singapore next to the database: gateway time p50 527 ms / p95 1,142 ms, client round trip p50 2.7 s / p95 4.0 s from my sandbox (before: p50 5.1 s / p95 6.4 s). On attack-like, opener-like or out-of-distribution traffic 9-27% of messages reach the 0.6B, so the 300 ms p95 does not hold there. The student is weaker on new attacks (out-of-distribution ASR 13.4% vs 6.6% for the 0.6B; the cascade gets 9.5%). Table and caveats: `reports/eval-v2.md`, ADR-0105. **Cold start:** about 13 s for the guard after 10 idle minutes (student 12 s, 0.6B 38 s on first escalation), about 22 s from page load to a playable boss 3 when both functions are cold.
- **In-distribution eval.** `gold_test` shares source datasets and generators with training (split by parent, near-duplicates checked). Gold labels are generation-time labels: a Gemini check agrees on 92.7% (kappa 0.851) and a human review is not done. The short-opener set (300 rows, LLM-written) is a regression detector, not proof of generalisation; Hindi openers sit at 15.0% FPR, exactly at their gate. A separate out-of-distribution set (934 rows from licence-checked public sets and an author/generator of a different model family, leakage-filtered; labels are LLM- or author-written, not human) gives the 0.6B an ASR of 6.6% [4.5, 9.1] and FPR of 17.3% [14.0, 20.7] (AUROC 0.959), so the headline 0.44% is an in-distribution number; the gold labels were also re-reviewed by three LLM judges (estimated 3.3% label noise), not by a human (`docs/gold-review.md`, ADR-0106). Cost: $0 actual, $0.11 per 1,000 requests at list prices (`docs/lanes/C-cost.md`, ADR-0107).
- **0% on templated attacks does not mean unbreakable.** Bosses 4 and 5 stopped every scripted and fresh attack in my runs; human players will find gaps. Boss 2 and 3 difficulty depends on which Groq model answers (free-tier token caps force fallback models), so the table above describes one day's vault chain.
- **Small samples.** v2 is one seed; v1 has three (ASR 0.49 +- 0.17%, FPR 1.83 +- 0.24%). A fourth v1 seed diverged in fp16 and is disclosed. DPO gave no significant gain (McNemar p = 1.0), so SFT-only ships.
- **Operations.** GitHub Actions is blocked by a billing lock on the owner's account, so workflows skip until `ACTIONS_BILLING_OK` is set and a local gate is used (ADR-0103); the canary and drift jobs also need repository secrets.

## Repo map
- `src/breachlab/gateway` game logic, leak detector, guards, FastAPI app; `train`, `serve`, `eval` training stages, ONNX runtime, metrics and reports; `flywheel`, `monitor` labelling, registry promote/rollback, canary, drift
- `web/` Next.js app; `vercel-guard/` the guard function; `space/` guard container image; `dashboards/` Grafana JSON; `configs/` levels, budget, gate
- `docs/` ADRs, [runbook](docs/runbook.md), [local dev](docs/LOCAL_DEV.md), [PRD gap](docs/prd-gap.md), lane reports; `reports/` eval reports; `docs/media/` demo and screenshots

## License
Code, model weights and model card: Apache-2.0 ([`LICENSE`](LICENSE)). Dataset (it includes share-alike rows from databricks-dolly-15k): CC-BY-SA-3.0 ([`DATASET_CARD.md`](DATASET_CARD.md), [`DATA_LICENSE.md`](DATA_LICENSE.md)). The model's training data includes CC-BY-SA-3.0 rows.
