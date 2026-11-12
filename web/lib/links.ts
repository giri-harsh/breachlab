// Public repository: giri-harsh/breachlab (ADR-0100). Public links come from build-time env so a rename/transfer (HF namespace, repo) needs no code change.
// scripts/link_check.py parses the `NAME = process.env.X ?? "default"` constants and the template URLs below.
export const HF_NAMESPACE = process.env.NEXT_PUBLIC_HF_NAMESPACE ?? "Pseudo-Harsh";
export const REPO = process.env.NEXT_PUBLIC_GITHUB_REPO ?? "giri-harsh/breachlab";
export const GUARD_MODEL_REPO = process.env.NEXT_PUBLIC_HF_MODEL ?? "breachlab-guard-v1";
export const DATA_REPO = process.env.NEXT_PUBLIC_HF_DATASET ?? "breachlab-data";
// The Space name is owned by the serving lane; the link is shown only once NEXT_PUBLIC_HF_SPACE is set.
export const SPACE_REPO = process.env.NEXT_PUBLIC_HF_SPACE ?? "";

export const HF_MODEL_URL = `https://huggingface.co/${HF_NAMESPACE}/${GUARD_MODEL_REPO}`;
export const HF_DATASET_URL = `https://huggingface.co/datasets/${HF_NAMESPACE}/${DATA_REPO}`;
export const HF_SPACE_URL: string | null = SPACE_REPO ? `https://huggingface.co/spaces/${HF_NAMESPACE}/${SPACE_REPO}` : null;
export const REPO_URL = `https://github.com/${REPO}`;
export const EVAL_REPORT_URL = `https://github.com/${REPO}/blob/main/reports/eval-v1.md`;
