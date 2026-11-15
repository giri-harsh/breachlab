# OOD test set card (`data/ood/ood_test.jsonl`)

An out-of-distribution test set for the BreachLab guard (v2 backlog item 5, lane O task L2). **Labels are LLM- or author-written, not human-verified.** It is committed to plain git (not DVC), split `ood_test`, same row schema as `data/processed/v2` (the sidecar `data/ood/ood_test.meta.jsonl` holds licence, origin and the screening verdict per row). It is for evaluation only and must never be added to training data.

**934 rows: 484 UNSAFE, 450 SAFE.** Regenerate with `scripts/build_ood.py` (stages `public`, `generate`, `screen`, `build`, `card`; cached stage files in `data/ood/stage/`). Score with `scripts/ood_report.py`.

## Sources and licences

| source | licence | origin | UNSAFE | SAFE |
|---|---|---|---|---|
| ood-authored-claude | CC0-1.0 (author-written) | authored | 150 | 146 |
| ood-generated-gemma-4-31b-it | CC0-1.0 (model-generated for this set) | generated | 142 | 118 |
| Lakera/mosscap_prompt_injection | MIT | public | 68 | 0 |
| OpenAssistant/oasst1 | Apache-2.0 | public | 0 | 77 |
| TrustAIRLab/in-the-wild-jailbreak-prompts | MIT | public | 19 | 46 |
| fka/prompts.chat | CC0-1.0 | public | 0 | 24 |
| neuralchemy/Prompt-injection-dataset | Apache-2.0 | public | 65 | 10 |
| reshabhs/SPML_Chatbot_Prompt_Injection | MIT | public | 40 | 29 |

Origins: `public` = sampled from a licence-checked public dataset (licence taken from the HF dataset card tag); `authored` = written by the lane author (a Claude model: a different family from the Qwen, gpt-oss and Gemini models that produced the training mutations); `generated` = produced by Gemma-4-31b-it (no Llama or Kimi model is served on the project's Groq key) with dataset-construction prompts, specs in `scripts/build_ood.py`.

Skipped sources (licence unclear, non-commercial, gated or out of scope):

- `xTRam1/safe-guard-prompt-injection`: no licence stated on the dataset card
- `imoxto/prompt_injection_cleaned_dataset`: no licence stated on the dataset card
- `geekyrakshit/prompt-injection-dataset`: no licence stated; aggregate of other sets (overlap with training sources)
- `hackaprompt/hackaprompt-dataset`: MIT but gated (terms not accepted for this token); hackaprompt rows also reach us via neuralchemy
- `qualifire/prompt-injections-benchmark`: CC-BY-NC-4.0 (non-commercial)
- `lmsys/toxic-chat`: CC-BY-NC-4.0 (non-commercial)
- `allenai/wildjailbreak`: ODC-BY but gated and model-generated; not requested
- `yanismiraoui/prompt_injections`: Apache-2.0 but 7 European languages outside the en/hi/hinglish schema

Nothing from ADR-0013 (deepset/prompt-injections, jackhhao/jailbreak-classification, Lakera/gandalf_ignore_instructions, JailbreakBench/JBB-Behaviors, ultrachat_200k, dolly-15k) is sampled; overlap through other sets is removed by the leakage filter.

## Counts

| language | UNSAFE | SAFE |
|---|---|---|
| en | 385 | 349 |
| hi | 47 | 50 |
| hinglish | 52 | 51 |

Hindi plus Hinglish: 99 UNSAFE and 101 SAFE rows (Devanagari and romanised).

UNSAFE categories: exfiltration 231, injection 160, jailbreak 93.

Attack families (top): mosscap_human 68, neuralchemy 65, spml_user_prompt 40, gen_hinglish 26, gen_hindi 26, gen_roleplay 21, wild_jailbreak 19, gen_multistep 19, gen_indirect 18, gen_fake_system 17, gen_casual 15, prompt_extraction 10, format_trick 10, authority_claim 10.

## Construction

1. Candidates: author-written rows (`scripts/ood_authored.py`: English casual, role-play, fake system messages, code-comment injection, translation tricks, multi-step, polite requests, Hindi and Hinglish; benign rows that mention rules, passwords, secrets, codes and security, greetings, meta questions about AI), public samples (seed 20261113, text up to 1200 characters, NSFW keyword filter) and Gemma-generated rows (script check: `hi` must contain Devanagari, `en` and `hinglish` must not).
2. Label screen (non-authored rows only): each public and generated row is judged by Gemini (`breachlab.eval.gold_judge`, the gold double-check prompt). Rows whose label the judge contradicts are dropped; rows with no source category take the judge's category. Dropped by this screen: Lakera/mosscap_prompt_injection:UNSAFE->SAFE 22, OpenAssistant/oasst1:SAFE->UNSAFE 2, TrustAIRLab/in-the-wild-jailbreak-prompts:SAFE->UNSAFE 4, TrustAIRLab/in-the-wild-jailbreak-prompts:UNSAFE->SAFE 14, fka/prompts.chat:SAFE->UNSAFE 1, neuralchemy/Prompt-injection-dataset:SAFE->UNSAFE 2, neuralchemy/Prompt-injection-dataset:UNSAFE->SAFE 13, ood-generated-gemma-4-31b-it:SAFE->UNSAFE 2, ood-generated-gemma-4-31b-it:UNSAFE->SAFE 1, reshabhs/SPML_Chatbot_Prompt_Injection:UNSAFE->SAFE 30. The screen removes mislabelled source rows (for example Mosscap prompts that are harmless chat, or SPML prompts that are attacks only relative to a system prompt) and makes the set somewhat easier than raw public data.
3. Leakage filter against 76,135 reference rows: exact (casefolded, whitespace-normalised) and MinHash near-duplicates at Jaccard >= 0.7 (3-character shingles, 128 permutations; the training dedup used 0.8) against v1 and v2 `train`, `val`, `gold_test`, `mining_pool`, v1 `seeds`, v2 `new_rows` and the opener slice. **58 candidate rows removed** (source, match kind: count):
   - OpenAssistant/oasst1, near: 1
   - TrustAIRLab/in-the-wild-jailbreak-prompts, exact: 44
   - TrustAIRLab/in-the-wild-jailbreak-prompts, near: 3
   - neuralchemy/Prompt-injection-dataset, exact: 2
   - ood-authored-claude, exact: 1
   - ood-authored-claude, near: 4
   - ood-generated-gemma-4-31b-it, near: 2
   - reshabhs/SPML_Chatbot_Prompt_Injection, near: 1
   Pools checked: `data/processed/v1/gold_test.jsonl` (1500), `data/processed/v1/mining_pool.jsonl` (5000), `data/processed/v1/seeds.jsonl` (15537), `data/processed/v1/train.jsonl` (19330), `data/processed/v1/val.jsonl` (2147), `data/processed/v2/gold_test.jsonl` (1500), `data/processed/v2/mining_pool.jsonl` (5000), `data/processed/v2/new_rows.jsonl` (2172), `data/processed/v2/train.jsonl` (21242), `data/processed/v2/val.jsonl` (2407), `data/review/opener_slice_v1.jsonl` (300). The removed rows are listed (text, source, matching pool) in `data/ood/stage/leakage_dropped.jsonl`.
4. Within-set near-duplicates removed: 1; the per-source cap dropped 30 rows so that no public source dominates.

## Caveats

- Labels are not human-verified. Author-written rows carry the author's intent; public rows carry the source label confirmed by one LLM judge; generated rows carry the requested label confirmed by one LLM judge. Expect a few percent label noise (`docs/gold-review.md` measures the same effect on gold_test).
- The author and the generator are LLMs: this measures robustness to new phrasing and new public distributions, not to real players. Real-player data (backlog item 5) remains open.
- Several public sources are compiled from earlier sets (neuralchemy includes HackAPrompt and other material; SPML overlaps Gandalf-style prompts). The leakage filter removes near-copies of anything in the training pools but cannot know what the base model saw in pretraining.
- Public attacks are not all vault-specific: in-the-wild jailbreaks target general chatbots, so category labels follow the judge or the source taxonomy.
- Hindi and Hinglish rows are author-written or Gemma-generated and were not reviewed by a native speaker.
- The class ratio is balanced by construction; it is not a base rate.
