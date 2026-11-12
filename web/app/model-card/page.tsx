import { loadScorecard, ms, num, pct, shipped } from "@/lib/scorecard";
import { EVAL_REPORT_URL, HF_DATASET_URL, HF_MODEL_URL, HF_SPACE_URL, REPO_URL } from "@/lib/links";

export const dynamic = "force-static";
export const metadata = { title: "Model card · BreachLab" };

export default function ModelCard() {
  const card = loadScorecard();
  const row = shipped(card);
  const space = HF_SPACE_URL;
  const links: [string, string][] = [
    ["Guard model and ONNX export (Hugging Face)", HF_MODEL_URL],
    ...(space ? ([["Guard service (Hugging Face Space)", space]] as [string, string][]) : []),
    ["Dataset (Hugging Face)", HF_DATASET_URL],
    ["Evaluation report", EVAL_REPORT_URL],
    ["Source code", REPO_URL],
  ];
  const metrics: [string, string][] = row
    ? [["Attack success (ASR)", pct(row[1].asr)], ["False positives (FPR)", pct(row[1].fpr)], ["F1", num(row[1].f1)], ["AUROC", num(row[1].auroc)], ["p95 latency", ms(row[1].p95_ms)]]
    : [];
  return (
    <div className="space-y-8">
      <section className="space-y-4">
        <h1 className="font-display text-xl leading-9 text-pink sm:text-3xl">MODEL CARD</h1>
        <p className="max-w-3xl text-lg text-soft">breachlab-guard-v1: Qwen3-0.6B + LoRA fine-tuned with SFT, temperature-calibrated, served on CPU. Research demo; not a claim of production safety.</p>
      </section>
      {row ? (
        <section aria-labelledby="mg" className="space-y-4">
          <h2 id="mg" className="font-display text-sm leading-7 text-yellow">Measured on gold_test ({card?.version})</h2>
          <dl className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5" data-testid="model-metrics">
            {metrics.map(([k, v]) => (
              <div key={k} className="rounded-[18px] border-[3px] border-border bg-surface p-4">
                <dt className="text-sm text-muted">{k}</dt>
                <dd className="mt-1 text-2xl font-bold text-cyan">{v}</dd>
              </div>
            ))}
          </dl>
          {typeof row[1].note === "string" && row[1].note && <p className="text-xs text-muted">{row[0]}: {row[1].note}</p>}
        </section>
      ) : <p className="text-muted" data-testid="no-scorecard">Scorecard not published yet.</p>}
      <section aria-labelledby="ln" className="space-y-3">
        <h2 id="ln" className="font-display text-sm text-cyan">LINKS</h2>
        <ul className="space-y-2">
          {links.map(([t, h]) => (
            <li key={h}><a className="inline-flex min-h-11 items-center underline" href={h}>{t}</a></li>
          ))}
        </ul>
        <p className="text-xs text-muted">Some links may be private until the public release.</p>
      </section>
    </div>
  );
}
