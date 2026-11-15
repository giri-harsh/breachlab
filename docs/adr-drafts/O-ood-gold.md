## ADR-DRAFT-O1 Out-of-distribution test set and LLM-consensus gold review (lane O, 2026-11-13)
Context: v2 backlog items (1) review of the 393 disagreeing gold rows and (5) a fresh out-of-distribution (OOD) test set.
No real-player traffic or human reviewer is available in this container, and there is no budget.

Decision:
1. `data/ood/ood_test.jsonl` (plain git, not DVC, split `ood_test`, schema unchanged except that `Split` gains `ood_test`) is built by
   `scripts/build_ood.py` from three sources none of which was seen in training: licence-checked public sets (MIT, Apache-2.0, CC0 only;
   sets with no stated licence, NC licences or gated access are skipped and listed in `docs/data/OOD_CARD.md`), rows written by the lane
   author (a Claude model, a different family from the Qwen / gpt-oss / Gemini generators of the training data), and rows generated with
   Gemma-4-31b-it (no Llama or Kimi model is served on the Groq key). Non-authored labels are screened by a Gemini judge and rows it
   contradicts are dropped; all rows are leakage-filtered (exact and MinHash at Jaccard 0.7, stricter than the 0.8 used for training)
   against v1 and v2 train, val, gold_test, mining_pool, seeds and the opener slice. Labels are LLM- or author-written, not human-verified.
2. `scripts/ood_report.py` scores logits files with the repo metrics (ASR, FPR, AUROC, seeded percentile bootstrap, 95%).
3. Gold review is `scripts/gold_review.py`: judge A (Groq qwen3.8-27b, strict JSON classifier), judge B (Gemini 3-flash-preview, rubric and
   reasoning), judge C (Groq gpt-oss-120b, tie-break on A/B splits). The 393-id list is not recoverable (Neon queue not readable by this lane,
   judge cache not versioned), so the review set is the 300-row sample plus a re-run Gemini screen's disagreements. **DONE-BY-DECISION:
   LLM-consensus review, not human.** `gold_test.jsonl` is never modified; relabels are a proposal in `docs/lanes/O-gold-review.json`.

Consequences: the OOD set measures robustness to new phrasing and new public distributions, not to real players (item 5 stays open for
live traffic). Judge labels share blind spots with the generators; the human review of the queue in Neon remains the reference.
