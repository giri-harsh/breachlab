"""OPTIONAL, UNUSED (ADR-0037: Modal is permanently out of the plan). Modal CPU app: DeBERTa prompt-injection baseline (protectai/deberta-v3-base-prompt-injection-v2,
Apache-2.0, verified ADR-0013) plus the regex filter. Not unit-tested (weights run on Modal only)."""

import modal

app = modal.App("breachlab-baselines")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers>=4.51", "numpy", "pydantic>=2.7")
    .add_local_python_source("breachlab")
)
MODEL = "protectai/deberta-v3-base-prompt-injection-v2"


@app.cls(image=image, cpu=8, memory=8192, scaledown_window=300)
class Baselines:
    @modal.enter()
    def load(self) -> None:
        from transformers import pipeline

        self.deberta = pipeline(
            "text-classification", model=MODEL, truncation=True, max_length=512, device=-1
        )

    @modal.method()
    def regex(self, texts: list[str]) -> list[dict[str, object]]:
        from breachlab.serve.baselines import regex_classify

        return [dict(zip(("label", "category"), regex_classify(t), strict=True)) for t in texts]

    @modal.method()
    def deberta_batch(self, texts: list[str]) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for r in self.deberta(texts, batch_size=16):
            unsafe = r["label"].upper() == "INJECTION"
            p = float(r["score"]) if unsafe else 1.0 - float(r["score"])
            out.append({"label": "UNSAFE" if unsafe else "SAFE", "p_unsafe": p})
        return out
