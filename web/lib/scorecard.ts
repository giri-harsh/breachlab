import fs from "node:fs";
import path from "node:path";

export type Metrics = Record<string, number | string | null>;
export type Card = { version: string; models: Record<string, Metrics> };

/** Reads public/scorecard.json at build/render time (server components only). Null when absent or malformed. */
export function loadScorecard(file = path.join(process.cwd(), "public", "scorecard.json")): Card | null {
  try {
    const raw = JSON.parse(fs.readFileSync(file, "utf-8")) as Card;
    if (typeof raw?.version !== "string" || typeof raw?.models !== "object" || raw.models === null) return null;
    return raw;
  } catch {
    return null;
  }
}

export const pct = (v: unknown, digits = 1) => (typeof v === "number" ? `${(v * 100).toFixed(digits)}%` : "-");
export const num = (v: unknown, digits = 3) => (typeof v === "number" ? v.toFixed(digits) : "-");
// scorecard.json keeps latency unrounded (ADR-0039); it is displayed as the true value rounded to the nearest ms, "n/a" when not measured.
export const ms = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? `${Math.round(v)} ms` : "n/a");

/** The shipped guard row: SFT-only, int4 graph as served on Vercel (ADR-0036, ADR-0082, guard v2 ADR-0094); override with NEXT_PUBLIC_SHIPPED_MODEL. */
export function shipped(card: Card | null): [string, Metrics] | null {
  if (!card) return null;
  const want = process.env.NEXT_PUBLIC_SHIPPED_MODEL ?? "qwen3-0.6b-sft-int4-v2";
  const m = card.models[want];
  return m ? [want, m] : null;
}
