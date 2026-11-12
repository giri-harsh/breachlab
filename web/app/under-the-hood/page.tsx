import { loadScorecard, ms, num, pct, shipped } from "@/lib/scorecard";
import { EVAL_REPORT_URL, HF_DATASET_URL, HF_MODEL_URL, HF_SPACE_URL, REPO_URL } from "@/lib/links";

export const dynamic = "force-static";
export const metadata = { title: "How it works · BreachLab" };

type Stage = { key: string; name: string; bosses: string; color: string; what: string; stats: [string, string][] };

/** Version history of the guard and the game (ADRs in docs/decisions.md). */
const TIMELINE: { when: string; title: string; body: string; adr: string; state: "done" | "now" | "next" }[] = [
  { when: "2026-10-30", title: "data-v1", body: "19,330 training rows (49.9% attacks, Hindi and Hinglish included), 1,500-row gold test set, leakage-checked splits.", adr: "ADR-0030", state: "done" },
  { when: "2026-10", title: "v1 gate: SFT 0.6B is champion", body: "LoRA SFT on Qwen3-0.6B beats every baseline; DPO adds nothing significant, so SFT-only ships.", adr: "ADR-0036", state: "done" },
  { when: "2026-11-01", title: "Guard live on a Vercel function", body: "4-bit weight-only ONNX graph on one CPU, its own threshold fitted on val for FPR at most 2%.", adr: "ADR-0082", state: "done" },
  { when: "2026-11-01", title: "Leak suite on the live guard", body: "Bosses 3 to 5: 0 leaks in 216 scripted attacks each; boss 1 leaked once in 210. 0% on templated attacks is not unbreakable: human players will find gaps.", adr: "ADR-0083", state: "done" },
  { when: "2026-11-11", title: "Cold start tamed", body: "The page wakes the guard on load; a booting guard answers 202 and the game waits for it.", adr: "ADR-0085", state: "done" },
  { when: "2026-11-11", title: "Base guard and opener gap measured", body: "The untuned base guard lets 99% of attacks through; short benign openers are the main false-positive gap.", adr: "ADR-0090", state: "done" },
  { when: "2026-11", title: "v1.1 Neon Arcade", body: "Bosses, points, hint tokens, skips and the verdict pill with the guard's attack probability.", adr: "ADR-0091", state: "now" },
  { when: "v2", title: "Guard v2: benign Hindi, Hinglish and short openers", body: "New benign training rows and a retrain to fix the false positives on short and Hindi messages.", adr: "ADR-0090", state: "next" },
];

export default function UnderTheHood() {
  const card = loadScorecard();
  const grafana = process.env.NEXT_PUBLIC_GRAFANA_SNAPSHOT_URL;
  const space = HF_SPACE_URL;
  const ship = shipped(card);
  const regex = card?.models.regex;
  const base = card?.models["qwen3-0.6b-zeroshot"];
  const stages: Stage[] = [
    { key: "regex", name: "Regex", bosses: "Bosses 2-5", color: "var(--yellow)", what: "Blocks obvious attack words before anything else runs.",
      stats: regex ? [["attacks passed", pct(regex.asr)], ["benign blocked", pct(regex.fpr)]] : [] },
    { key: "guard", name: "Input guard", bosses: "Bosses 3-5", color: "var(--pink)", what: "Qwen3-0.6B + LoRA scores every message in one forward pass (int4, CPU).",
      stats: ship ? [["attacks passed", pct(ship[1].asr)], ["benign blocked", pct(ship[1].fpr)], ["p95 model time", ms(ship[1].p95_ms)], ...(base ? ([["untuned base passed", pct(base.asr)]] as [string, string][]) : [])] : [] },
    { key: "vault", name: "Vault agent", bosses: "All bosses", color: "var(--cyan)", what: "A Groq-hosted open LLM holds the boss's secret code and tries to keep it.", stats: [["secret", "8 chars, per player"]] },
    { key: "filter", name: "Output filter", bosses: "Boss 5", color: "var(--green)", what: "Scans the reply and scrubs it if the code is in there, in any encoding.", stats: [["fails", "closed"]] },
    { key: "leak", name: "Leak detector", bosses: "Every reply", color: "var(--text)", what: "Deterministic check: base64, hex, ROT13, leetspeak, reversed and spaced-out codes all count as a win.", stats: [["label", "exact, no model"]] },
  ];
  return (
    <div className="space-y-14">
      <section className="space-y-4">
        <h1 className="font-display text-xl leading-9 text-cyan sm:text-3xl">UNDER THE HOOD</h1>
        <p className="max-w-3xl text-lg text-soft">A small guard model (Qwen3-0.6B, LoRA fine-tuned) classifies every message as safe, injection, jailbreak or exfiltration in one forward pass. Every breach is detected deterministically, so the game produces correctly labeled training data.</p>
        <p className="max-w-3xl text-sm text-muted" data-testid="l3-note">
          L3: same model, relaxed threshold. The Watcher runs the same guard as bosses 4 and 5, but it only blocks a message when it is nearly certain (threshold 0.99990224 on its attack score), so more attacks get through.
        </p>
      </section>

      <section aria-labelledby="path" className="space-y-5">
        <h2 id="path" className="font-display text-sm text-yellow sm:text-base">REQUEST PATH</h2>
        <p className="text-sm text-muted">Every message travels left to right (top to bottom on a phone). Numbers are from the gold test set{card ? ` (${card.version})` : ""}.</p>
        <div className="relative" data-testid="request-path">
          <div aria-hidden className="absolute bottom-6 left-[22px] top-6 w-1 rounded bg-raised md:bottom-auto md:left-6 md:right-6 md:top-[22px] md:h-1 md:w-auto">
            <span className="absolute left-[-5px] top-0 h-3.5 w-3.5 rounded-full bg-yellow [animation:travel-y_6s_linear_infinite] md:left-0 md:top-[-5px] md:[animation:travel-x_6s_linear_infinite]" />
          </div>
          <ol className="relative grid gap-4 md:grid-cols-5">
            {stages.map((s, i) => (
              <li key={s.key} className="flex gap-4 md:flex-col md:gap-3" data-testid={`stage-${s.key}`}>
                <span className="anim-glow relative z-10 flex h-12 w-12 shrink-0 items-center justify-center rounded-[14px] border-[3px] bg-bg font-display text-xs"
                  style={{ borderColor: s.color, color: s.color, animationDelay: `${i * 1.2}s` }} aria-hidden>{i + 1}</span>
                <div className="flex-1 rounded-[18px] border-[3px] border-border bg-surface p-4">
                  <h3 className="font-bold" style={{ color: s.color }}>{s.name}</h3>
                  <p className="text-xs text-muted">{s.bosses}</p>
                  <p className="mt-2 text-sm text-soft">{s.what}</p>
                  {s.stats.length > 0 && (
                    <dl className="mt-3 space-y-1 text-sm">
                      {s.stats.map(([k, v]) => (<div key={k} className="flex justify-between gap-2"><dt className="text-muted">{k}</dt><dd className="font-bold">{v}</dd></div>))}
                    </dl>
                  )}
                </div>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section aria-labelledby="sc" className="space-y-4">
        <h2 id="sc" className="font-display text-sm leading-7 text-yellow sm:text-base">Scorecard {card ? `(${card.version}, gold_test)` : ""}</h2>
        {card ? (
          <div>
            <div className="overflow-x-auto rounded-[20px] border-[3px] border-border" tabIndex={0} role="region" aria-label="Scorecard table, scrolls sideways on small screens">
              <table className="w-full min-w-[34rem] text-left text-sm">
                <caption className="sr-only">Guard comparison on the held-out gold test set</caption>
                <thead className="bg-raised text-cyan"><tr>{["Model", "ASR", "FPR", "F1", "AUROC", "p95 latency"].map((h) => <th key={h} scope="col" className="px-3 py-3 font-bold">{h}</th>)}</tr></thead>
                <tbody>
                  {Object.entries(card.models).map(([name, m]) => (
                    <tr key={name} className={`border-t-2 border-raised ${ship?.[0] === name ? "bg-surface text-yellow" : ""}`}>
                      <th scope="row" className="px-3 py-2 font-normal">{name}{ship?.[0] === name ? " (shipped)" : ""}</th>
                      <td className="px-3">{pct(m.asr)}</td><td className="px-3">{pct(m.fpr)}</td><td className="px-3">{num(m.f1)}</td><td className="px-3">{num(m.auroc)}</td><td className="px-3">{ms(m.p95_ms)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-2 text-xs text-muted">ASR = share of attacks passed (lower is better). FPR = share of benign prompts blocked. In-distribution numbers; see the evaluation report for caveats.</p>
            <div data-testid="latency-note" className="mt-4 rounded-[18px] border-[3px] border-border bg-surface p-4 text-sm leading-6 text-soft">
              <p className="font-bold text-text">How latency was measured</p>
              <p className="mt-1">
                p95 is the 95th percentile of model time per request (one request at a time, messages capped at 500 characters), shown rounded to the nearest millisecond; n/a means the row was not measured.
                {ship && typeof ship[1].p95_ms === "number" && <> The shipped guard ({ship[0]}) measured <span data-testid="shipped-p95">{ms(ship[1].p95_ms)}</span> on the real host (a Vercel function with one vCPU, warm, one request at a time), so the 300 ms target is <strong>not met</strong>. A request after a long idle takes about 13 s while the instance boots.</>}
                {" "}Rows measured on other hardware are not comparable: the 1.7B row used a 4-vCPU Kaggle CPU with an fp32 ONNX export at 512 tokens, and it is not the shipped configuration. API models (regex, Prompt Guard, Gemini) have no comparable latency. Numbers come from shared hardware and are noisy.
              </p>
            </div>
          </div>
        ) : <p className="text-muted" data-testid="no-scorecard">Scorecard not published yet.</p>}
      </section>

      <section aria-labelledby="tl" className="space-y-5">
        <h2 id="tl" className="font-display text-sm text-pink sm:text-base">VERSION TIMELINE</h2>
        <ol className="relative space-y-4 border-l-4 border-raised pl-6" data-testid="timeline">
          {TIMELINE.map((t) => (
            <li key={t.title} className="relative">
              <span aria-hidden className={`absolute -left-[34px] top-1.5 h-4 w-4 rounded-full border-[3px] ${t.state === "done" ? "border-green bg-green" : t.state === "now" ? "anim-glow border-cyan bg-cyan" : "border-muted-2 bg-bg"}`} />
              <p className="font-display text-[10px] leading-5 text-muted">{t.when} · {t.state === "done" ? "SHIPPED" : t.state === "now" ? "NOW" : "PLANNED"}</p>
              <h3 className="font-bold">{t.title}</h3>
              <p className="text-sm text-soft">{t.body} <span className="text-muted">({t.adr})</span></p>
            </li>
          ))}
        </ol>
      </section>

      <section aria-labelledby="fw" className="space-y-3">
        <h2 id="fw" className="font-display text-sm text-cyan sm:text-base">THE FLYWHEEL</h2>
        <pre className="overflow-x-auto rounded-[18px] border-[3px] border-border bg-surface p-4 text-xs leading-5 text-soft">{`capture -> label (leak / judge / human) -> build dataset -> train (SFT, mining, DPO)
   ^                                                                |
   |                                                                v
 shadow 48h <- promote (alias champion) <- eval gate <-------- challenger`}</pre>
      </section>

      <section aria-labelledby="oa" className="space-y-3">
        <h2 id="oa" className="font-display text-sm text-green sm:text-base">OPEN ARTIFACTS</h2>
        <ul className="list-disc space-y-2 pl-5">
          <li><a className="underline" href={HF_MODEL_URL}>Guard model (Hugging Face)</a></li>
          {space && <li><a className="underline" href={space}>Guard service (Hugging Face Space)</a></li>}
          <li><a className="underline" href={HF_DATASET_URL}>Dataset (Hugging Face)</a></li>
          <li><a className="underline" href={EVAL_REPORT_URL}>Evaluation report</a></li>
          <li><a className="underline" href={REPO_URL}>Source code (GitHub)</a></li>
        </ul>
        {grafana ? (<iframe src={grafana} title="Live metrics" className="h-96 w-full rounded-[18px]" />) : <p className="text-sm text-muted">Live metrics snapshot: not published yet.</p>}
      </section>
    </div>
  );
}
