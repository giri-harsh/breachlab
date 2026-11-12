// Share-card inputs (CONTRACTS.md 7.9), shared by /api/og, /share, ShareCard and /daily. Lane V.
// Every value that reaches the image or a URL goes through `parseCard`, so bad input is clamped, never echoed.

export const BOSSES: Record<number, string> = {
  1: "Chatty Bot",
  2: "Word Cop",
  3: "The Watcher",
  4: "Trained Sentinel",
  5: "Double Lock",
  6: "Daily Vault",
};

// Hex values of the arcade palette (docs/design/arcade-reference.html). The OG image cannot read CSS variables;
// components use `v("pink")`, which prefers lane U's variable and falls back to the same hex.
export const HEX = {
  bg: "#12062B",
  surface: "#1E0B45",
  raised: "#2E1566",
  border: "#4A2A8A",
  grid: "#2A1259",
  pink: "#FF3EA5",
  cyan: "#3EF0FF",
  yellow: "#FFE14D",
  green: "#5BFF8A",
  text: "#F2EAFF",
  muted: "#B9A6E8",
  "muted-2": "#8A74C4",
  ink: "#2A0A3F",
} as const;
export type Swatch = keyof typeof HEX;
export const v = (name: Swatch): string => (name === "ink" || name === "grid" ? HEX[name] : `var(--${name}, ${HEX[name]})`);

export type Card = { boss: number; msgs: number; pts: number; name: string };
export const DEFAULT_NAME = "PLAYER 1";

type Raw = string | string[] | null | undefined;
const first = (x: Raw): string => (Array.isArray(x) ? (x[0] ?? "") : (x ?? ""));

function clampInt(x: Raw, lo: number, hi: number, fallback: number): number {
  const s = first(x).trim();
  if (!/^-?\d{1,9}$/.test(s)) return fallback;
  return Math.min(hi, Math.max(lo, Number.parseInt(s, 10)));
}

/** Letters, digits, space and `_ . -` only (what the pixel font can draw), single spaces, at most 20 chars. */
export function cleanName(x: Raw): string {
  const s = first(x)
    .normalize("NFKD")
    .replace(/[^A-Za-z0-9 _.-]/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 20)
    .trim();
  return s || DEFAULT_NAME;
}

export function parseCard(q: { get(k: string): string | null } | Record<string, Raw>): Card {
  const get = (k: string): Raw => (typeof (q as { get?: unknown }).get === "function" ? (q as { get(k: string): string | null }).get(k) : (q as Record<string, Raw>)[k]);
  return {
    boss: clampInt(get("boss"), 1, 6, 1),
    msgs: clampInt(get("msgs"), 1, 20, 1),
    pts: clampInt(get("pts"), 0, 99999, 0),
    name: cleanName(get("name")),
  };
}

export function cardQuery(c: Card): string {
  const p = parseCard({ boss: String(c.boss), msgs: String(c.msgs), pts: String(c.pts), name: c.name });
  return new URLSearchParams({ boss: String(p.boss), msgs: String(p.msgs), pts: String(p.pts), name: p.name }).toString();
}

export function headline(c: Card): string {
  return `I BEAT ${BOSSES[c.boss].toUpperCase()} IN ${c.msgs} ${c.msgs === 1 ? "MESSAGE" : "MESSAGES"}`;
}

export const formatPts = (n: number): string => n.toLocaleString("en-US");
