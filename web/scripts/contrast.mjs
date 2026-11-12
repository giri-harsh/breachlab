// WCAG 2.x contrast ratios for every text/background pair the Neon Arcade UI uses. Run: node scripts/contrast.mjs
// AA: >= 4.5 for normal text, >= 3 for large text (>= 24px, or >= 18.66px bold) and for non-text UI (borders, icons).
// Exits 1 when a pair in use fails. "-orig" rows document the design shade that was replaced.
const C = {
  bg: "#12062B", surface: "#1E0B45", raised: "#2E1566", border: "#4A2A8A",
  pink: "#FF3EA5", cyan: "#3EF0FF", yellow: "#FFE14D", green: "#5BFF8A",
  text: "#F2EAFF", muted: "#B9A6E8", "muted-2": "#A08BDB", "muted-2-orig": "#8A74C4",
  ink: "#2A0A3F", "ink-cyan": "#10213A", soft: "#D7C8FF", "chip-grey": "#C9C2D9", "chip-grey-bg": "#3A3150",
  "border-ui": "#7A5BC7", "border-orig": "#4A2A8A",
};
const lum = (hex) => {
  const v = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255).map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
};
export const ratio = (a, b) => { const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); };
const fg = ["text", "muted", "muted-2", "soft", "pink", "cyan", "yellow", "green"];
const pairs = [
  ...fg.flatMap((f) => ["bg", "surface", "raised"].map((b) => [f, b, "text"])),
  ["muted-2-orig", "bg", "text"], ["muted-2-orig", "surface", "text"], ["muted-2-orig", "raised", "text"],
  ["ink", "yellow", "text"], ["ink", "pink", "text"], ["ink", "green", "text"], ["ink", "cyan", "text"], ["ink-cyan", "cyan", "text"],
  ["chip-grey", "chip-grey-bg", "text"],
  ["border-ui", "bg", "ui"], ["border-ui", "surface", "ui"], ["border-orig", "bg", "decor"],
  ["pink", "bg", "ui"], ["cyan", "bg", "ui"], ["yellow", "bg", "ui"],
];
console.log("| foreground | background | ratio | AA needs | result |\n|---|---|---|---|---|");
let fails = 0;
for (const [f, b, kind] of pairs) {
  const r = ratio(C[f], C[b]);
  const need = kind === "text" ? 4.5 : 3;
  const ok = r >= need;
  const inUse = !f.endsWith("-orig");
  if (!ok && inUse) fails++;
  const label = kind === "text" ? "4.5 (text)" : kind === "ui" ? "3 (UI border/icon)" : "3 (decorative only)";
  console.log(`| ${f} ${C[f]} | ${b} ${C[b]} | ${r.toFixed(2)} | ${label} | ${ok ? "PASS" : inUse ? "FAIL" : "FAIL (replaced, not used for text)"} |`);
}
console.log(`\nfailing pairs in use: ${fails}`);
process.exitCode = fails ? 1 : 0;
