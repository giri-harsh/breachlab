// GET /api/og?boss=1-6&msgs=1-20&pts=0-99999&name=<=20 chars -> 1200x630 PNG share card (CONTRACTS.md 7.9). Lane V.
// The pixel font is bundled next to this file (SIL OFL, see OFL.txt) so the route needs no network.
import { ImageResponse } from "next/og";
import { BOSSES, HEX, formatPts, parseCard } from "./card";

export const runtime = "edge";

const W = 1200;
const H = 630;
const CELL = 48;

let font: Promise<ArrayBuffer> | null = null;
const loadFont = () => (font ??= fetch(new URL("./PressStart2P-Regular.ttf", import.meta.url)).then((r) => r.arrayBuffer()));

export async function GET(req: Request) {
  const c = parseCard(new URL(req.url).searchParams);
  const boss = BOSSES[c.boss].toUpperCase();
  const msgLine = `IN ${c.msgs} ${c.msgs === 1 ? "MESSAGE" : "MESSAGES"}`;
  const bossSize = boss.length > 12 ? 56 : 68;
  const lines = [
    ...Array.from({ length: Math.floor(W / CELL) }, (_, i) => ({ left: (i + 1) * CELL, top: 0, width: 2, height: H })),
    ...Array.from({ length: Math.floor(H / CELL) }, (_, i) => ({ left: 0, top: (i + 1) * CELL, width: W, height: 2 })),
  ];
  return new ImageResponse(
    (
      <div style={{ width: W, height: H, display: "flex", background: HEX.bg, padding: 24, fontFamily: "PressStart2P" }}>
        <div style={{ display: "flex", flex: 1, border: `6px solid ${HEX.cyan}`, borderRadius: 40, padding: 6 }}>
          <div
            style={{
              display: "flex", flex: 1, position: "relative", overflow: "hidden", background: HEX.bg,
              border: `6px solid ${HEX.pink}`, borderRadius: 30, flexDirection: "column", alignItems: "center",
              justifyContent: "space-between", padding: "40px 48px",
            }}
          >
            {lines.map((l, i) => (
              <div key={i} style={{ position: "absolute", background: HEX.grid, ...l }} />
            ))}
            <div style={{ display: "flex", width: "100%", justifyContent: "space-between", fontSize: 22 }}>
              <div style={{ display: "flex", color: HEX.pink }}>
                BREACH<span style={{ color: HEX.cyan }}>LAB</span>
              </div>
              <div style={{ display: "flex", color: HEX.cyan }}>{`BOSS ${c.boss}`}</div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", color: HEX.yellow, textAlign: "center" }}>
              <div style={{ display: "flex", fontSize: 40, textShadow: `4px 4px 0 ${HEX.pink}` }}>I BEAT</div>
              <div style={{ display: "flex", fontSize: bossSize, marginTop: 28, textShadow: `6px 6px 0 ${HEX.pink}` }}>{boss}</div>
              <div style={{ display: "flex", fontSize: 40, marginTop: 28, textShadow: `4px 4px 0 ${HEX.pink}` }}>{msgLine}</div>
            </div>
            <div style={{ display: "flex", width: "100%", justifyContent: "space-between", fontSize: 24 }}>
              <div style={{ display: "flex", color: HEX.yellow }}>{`SCORE ${formatPts(c.pts)}`}</div>
              <div style={{ display: "flex", color: HEX.text }}>{c.name.toUpperCase()}</div>
            </div>
          </div>
        </div>
      </div>
    ),
    {
      width: W,
      height: H,
      fonts: [{ name: "PressStart2P", data: await loadFont(), style: "normal", weight: 400 }],
      headers: { "cache-control": "public, max-age=86400, s-maxage=86400, immutable" },
    },
  );
}
