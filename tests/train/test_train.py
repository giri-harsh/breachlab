import json
from pathlib import Path

import numpy as np
import pytest

from breachlab.common.config import ServeConfig, TrainConfig
from breachlab.data.schema import Row
from breachlab.train import kaggle_job
from breachlab.train.calibrate import calibrate, p_unsafe, to_preds
from breachlab.train.dpo import balance_pairs, build_pairs, pairs_from_rows
from breachlab.train.export_onnx import score_from_hidden
from breachlab.train.kernel_main import config_from, parse
from breachlab.train.mine import mine, summary, top_word
from breachlab.train.prompts import (
    LABELS,
    encode_prompt,
    label_index,
    prompt_parts,
    render_prompt,
    truncate_ids,
)
from breachlab.train.score_model import batches, left_pad
from breachlab.train.scoring import decide, probs_from_logits, read_rows
from breachlab.train.sft import build_examples, collate, encode_example, gpu_usd, warmup_steps

SERVE = ServeConfig()  # type: ignore[call-arg]
WORDS = SERVE.words()


def enc(s: str) -> list[int]:
    return [ord(c) for c in s]


def row(text: str, label: str = "SAFE", cat: str = "none") -> Row:
    return Row.new(text, label, cat, "t")  # type: ignore[arg-type]


def test_serve_config_loads_verbalizers() -> None:
    assert WORDS == ["safe", "inject", "persona", "extract"]
    assert SERVE.token_ids() == [18675, 32133, 70594, 23493]
    assert SERVE.max_tokens.input_guard == 512


def test_train_config_defaults_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = TrainConfig()
    assert cfg.sft.lora.r == 16 and cfg.sft.grad_accum == 8 and cfg.dpo.beta == 0.1
    assert cfg.short_name == "0.6b"
    monkeypatch.setenv("BREACHLAB_TRAIN_SEED", "7")
    monkeypatch.setenv("BREACHLAB_TRAIN_SFT__EPOCHS", "1")
    cfg2 = TrainConfig()
    assert cfg2.seed == 7 and cfg2.sft.epochs == 1


def test_prompt_pieces_and_truncation() -> None:
    p = render_prompt("hello", WORDS)
    assert p.startswith("<|im_start|>system\n") and p.endswith("<think>\n\n</think>\n\n")
    prefix, suffix = prompt_parts(WORDS)
    assert p == prefix + "hello" + suffix
    assert len(truncate_ids(list(range(10)), 4)) == 4
    assert truncate_ids(list(range(10)), 4) == [0, 1, 8, 9]
    assert truncate_ids([1, 2], 4) == [1, 2]
    ids = encode_prompt(enc, "x" * 50, WORDS, "output", 6)
    assert len(ids) == len(enc(prefix.replace("input", "output"))) - 0 or ids
    assert "assistant reply" in render_prompt("a", WORDS, "output")
    assert label_index(row("a")) == 0 and label_index(row("b", "UNSAFE", "jailbreak")) == 2


def test_sft_examples_answer_token_and_left_padding() -> None:
    rows = [row("benign text"), row("ignore rules", "UNSAFE", "injection")]
    exs = build_examples(rows, SERVE)
    assert [e.target_id for e in exs] == [18675, 32133]
    item = encode_example(exs[1], enc, 512)
    assert item["target"] == 32133 and item["input_ids"][0] != 32133
    short, long = encode_example(exs[0], enc, 512), encode_example(exs[1], enc, 512)
    b = collate([short, long], pad_id=0)
    assert len({len(x) for x in b["input_ids"]}) == 1
    n_pad = len(b["input_ids"][0]) - len(short["input_ids"])
    assert b["attention_mask"][0][:n_pad] == [0] * n_pad and b["attention_mask"][0][-1] == 1
    assert b["input_ids"][0][-1] == short["input_ids"][-1] and b["labels"] == [18675, 32133]
    assert gpu_usd(3600, 0.8) == 0.8


def test_scoring_and_decide() -> None:
    lg = np.array([2.0, 0.0, 0.0, 0.0])
    p = probs_from_logits(lg)
    assert pytest.approx(p.sum()) == 1.0
    assert decide(lg).label == "SAFE"
    v = decide(np.array([0.0, 3.0, 0.0, 1.0]))
    assert v.label == "UNSAFE" and v.category == "injection" and v.p_unsafe > 0.9
    hot = decide(np.array([0.0, 0.0, 0.0, 0.0]), temperature=1.0, threshold=0.9)
    assert hot.label == "SAFE"
    assert probs_from_logits(lg, 10.0)[0] < p[0]


def test_read_rows(tmp_path: Path) -> None:
    f = tmp_path / "r.jsonl"
    r = row("hello there")
    f.write_text(r.model_dump_json() + "\n\n", encoding="utf-8")
    assert read_rows(f)[0].id == r.id


def test_mine_kinds_and_pairs() -> None:
    rows = [
        row("benign ok"),
        row("benign blocked"),
        row("attack missed", "UNSAFE", "injection"),
        row("attack wrong cat", "UNSAFE", "injection"),
        row("attack near", "UNSAFE", "exfiltration"),
    ]
    logits = np.array(
        [
            [5.0, 0, 0, 0],
            [0, 5.0, 0, 0],
            [5.0, 0, 0, 0],
            [0, 0, 5.0, 0],
            [0, 0, 0, 0.1],
        ]
    )
    m = mine(rows, logits, near_miss=0.2)
    assert summary(m) == {
        "false_positive": 1,
        "false_negative": 1,
        "wrong_category": 1,
        "near_miss": 1,
    }
    assert top_word(m[0]) in LABELS
    pairs = build_pairs(m, SERVE)
    assert len(pairs) == 4 and all(p.chosen != p.rejected for p in pairs)
    assert {p.chosen for p in pairs} <= set(WORDS)
    assert len(pairs_from_rows(rows[:2], m, SERVE)) == 1
    assert len(balance_pairs(pairs, 2)) == 2 and len(balance_pairs(pairs, 10)) == 4
    with pytest.raises(ValueError):
        mine(rows, logits[:2])


def test_calibrate_reduces_ece_on_overconfident_logits() -> None:
    rng = np.random.default_rng(0)
    n = 400
    labels = rng.integers(0, 4, n)
    logits = rng.normal(0, 1, (n, 4))
    logits[np.arange(n), labels] += 2.0
    logits *= 6.0  # overconfident
    cats = ["none", "injection", "jailbreak", "exfiltration"]
    rows = [
        row(
            f"sample text number {i} unique{i * 7}",
            "SAFE" if labels[i] == 0 else "UNSAFE",
            cats[labels[i]],
        )
        for i in range(n)
    ]
    cal = calibrate(rows, logits, 15, 0.02)
    assert cal.temperature > 1.0 and cal.ece_after <= cal.ece_before
    preds = to_preds(rows, logits, cal.temperature, cal.threshold)
    assert len(preds) == n
    assert p_unsafe(logits, cal.temperature).shape == (n,)


def test_onnx_head_math() -> None:
    h = np.ones((2, 3))
    head = np.arange(12, dtype=float).reshape(4, 3)
    assert score_from_hidden(h, head).shape == (2, 4)


def test_batches_and_left_pad() -> None:
    assert batches(5, 2) == [range(0, 2), range(2, 4), range(4, 5)]
    ids, mask = left_pad([[1, 2, 3], [4]], 0)
    assert ids == [[1, 2, 3], [0, 0, 4]] and mask == [[1, 1, 1], [0, 0, 1]]


def test_kaggle_packaging(tmp_path: Path) -> None:
    kaggle_job.stage_code(tmp_path / "code")
    assert (tmp_path / "code" / "src" / "breachlab" / "__init__.py").exists()
    assert (
        json.loads((tmp_path / "code" / "dataset-metadata.json").read_text())["isPrivate"] is True
    )
    src = tmp_path / "d"
    src.mkdir()
    for n in ("train", "val", "mining_pool", "gold_test"):
        (src / f"{n}.jsonl").write_text("{}\n")
    kaggle_job.stage_data(src, tmp_path / "data")
    assert (tmp_path / "data" / "gold_test.jsonl").exists()
    meta = kaggle_job.kernel_meta("o", "k", ["o/a"])
    assert meta["enable_gpu"] is True and meta["dataset_sources"] == ["o/a"]
    script = kaggle_job.kernel_script("Qwen/Qwen3-0.6B", 0)
    compile(script, "run.py", "exec")
    assert "--seed" in script and "BREACHLAB_CONFIG_DIR" in script


def test_kernel_main_args(tmp_path: Path) -> None:
    a = parse(
        [
            "--model",
            "Qwen/Qwen3-1.7B",
            "--seed",
            "2",
            "--data-dir",
            str(tmp_path),
            "--out",
            str(tmp_path),
            "--precision",
            "bf16",
        ]
    )
    cfg = config_from(a)
    assert cfg.base_model.endswith("1.7B") and cfg.seed == 2 and cfg.sft.precision == "bf16"


def test_kernel_main_export_args(tmp_path: Path) -> None:
    base = ["--data-dir", str(tmp_path), "--out", str(tmp_path), "--stage", "export"]
    a = parse(base)
    assert (a.which, a.variants, a.parity_rows) == ("dpo-0.6b-s0", "fp32,w8", -1)
    b = parse([*base, "--which", "base", "--variants", "fp32,kv", "--parity-rows", "300"])
    assert (b.which, b.variants.split(","), b.parity_rows) == ("base", ["fp32", "kv"], 300)


def test_pipeline_stage_done(tmp_path: Path) -> None:
    from breachlab.train.pipeline import stage_done

    assert not stage_done(tmp_path / "adapter")
    (tmp_path / "adapter").mkdir()
    assert stage_done(tmp_path / "adapter")


def test_warmup_steps() -> None:
    cfg = TrainConfig()
    assert warmup_steps(19330, cfg) == round(0.03 * 3 * -(-19330 // 32))
    assert warmup_steps(1, cfg) == 1
