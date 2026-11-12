"""Check the four verbalizers are single tokens in Qwen3 tokenizers; write configs/serve.yaml.

Downloads tokenizer files ONLY (never *.safetensors / *.bin).
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

MODELS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"]
WORDS = ["safe", "injection", "jailbreak", "exfiltration"]
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
FALLBACKS = {
    "safe": ["safe", "ok", "benign"],
    "injection": ["injection", "inject", "override"],
    "jailbreak": ["jailbreak", "persona", "jail"],
    "exfiltration": ["exfiltration", "extract", "leak"],
}
OUT = Path(__file__).resolve().parents[1] / "configs" / "serve.yaml"


def single_token_ids(tok, word: str) -> dict[str, int | None]:  # type: ignore[no-untyped-def]
    """Return {form: id or None} for the bare and leading-space forms."""
    out: dict[str, int | None] = {}
    for form, text in (("bare", word), ("space", " " + word)):
        ids = tok.encode(text, add_special_tokens=False).ids
        out[form] = ids[0] if len(ids) == 1 else None
    return out


def choose(tok) -> dict[str, dict[str, object]]:  # type: ignore[no-untyped-def]
    """Pick one consistent form for all labels, preferring bare (Qwen3 answers start after a newline)."""
    for form in ("bare", "space"):
        mapping: dict[str, dict[str, object]] = {}
        used: set[int] = set()
        for label, candidates in FALLBACKS.items():
            for cand in candidates:
                tid = single_token_ids(tok, cand)[form]
                if tid is not None and tid not in used:
                    used.add(tid)
                    mapping[label] = {
                        "word": cand if form == "bare" else " " + cand,
                        "form": form,
                        "id": tid,
                    }
                    break
        if len(mapping) == len(FALLBACKS):
            return mapping
    raise RuntimeError("no consistent single-token verbalizer set found")


def main() -> int:
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer

    results: dict[str, dict[str, dict[str, object]]] = {}
    for repo in MODELS:
        path = snapshot_download(repo, allow_patterns=TOKENIZER_FILES)
        tok = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
        results[repo] = choose(tok)
        for word in WORDS:
            print(repo, word, single_token_ids(tok, word))
    # the 0.6B and 1.7B share a vocabulary; require identical ids so one config serves both
    a, b = (results[m] for m in MODELS)
    same = all(a[k]["id"] == b[k]["id"] for k in a)
    print("identical mapping across sizes:", same)
    OUT.write_text(
        yaml.safe_dump(
            {
                "models": MODELS,
                "verbalizers": {k: v for k, v in results[MODELS[0]].items()},
                "identical_across_models": same,
                "max_tokens": {"input_guard": 512, "output_filter": 384},
                "threshold": None,  # tuned on val in Phase 3
                "temperature": None,  # fitted in Phase 3
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
