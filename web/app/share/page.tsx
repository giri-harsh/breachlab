// /share?boss=&msgs=&pts=&name= : the page a shared result links to (CONTRACTS.md 7.9). Lane V.
// og:image / twitter:image point at /api/og with the same (clamped) params; the page itself is a small arcade CTA.
import type { Metadata } from "next";
import { headers } from "next/headers";
import Link from "next/link";
import ShareCard from "@/components/ShareCard";
import StatsTicker from "@/components/StatsTicker";
import { cardQuery, formatPts, headline, parseCard, v } from "@/app/api/og/card";

type Search = Promise<Record<string, string | string[] | undefined>>;

async function origin(): Promise<string> {
  try {
    const h = await headers();
    const host = h.get("x-forwarded-host") ?? h.get("host");
    if (host && /^[A-Za-z0-9.:-]+$/.test(host)) {
      const proto = h.get("x-forwarded-proto") ?? (host.startsWith("localhost") || host.startsWith("127.") ? "http" : "https");
      return `${proto === "http" ? "http" : "https"}://${host}`;
    }
  } catch { /* fall through */ }
  return "https://breach.harshgiri.site";
}

export async function generateMetadata({ searchParams }: { searchParams: Search }): Promise<Metadata> {
  const card = parseCard(await searchParams);
  const base = await origin();
  const image = `${base}/api/og?${cardQuery(card)}`;
  const title = `${headline(card)} | BreachLab`;
  const description = `${card.name} scored ${formatPts(card.pts)} points tricking a guarded AI vault. Your turn.`;
  const alt = `${headline(card)}. Score ${formatPts(card.pts)}. Player ${card.name}.`;
  return {
    title,
    description,
    openGraph: { title, description, type: "website", url: `${base}/share?${cardQuery(card)}`, images: [{ url: image, width: 1200, height: 630, alt }] },
    twitter: { card: "summary_large_image", title, description, images: [{ url: image, alt }] },
  };
}

const cta: React.CSSProperties = {
  minHeight: 60, padding: "0 28px", borderRadius: 14, display: "inline-flex", alignItems: "center", gap: 12,
  fontSize: 14, textDecoration: "none",
};

export default async function SharePage({ searchParams }: { searchParams: Search }) {
  const card = parseCard(await searchParams);
  const q = cardQuery(card);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24, color: v("text") }}>
      <section
        style={{
          border: `4px solid ${v("pink")}`, borderRadius: 28, padding: "40px 20px", textAlign: "center",
          background: v("bg"), boxShadow: `0 0 0 4px ${v("bg")}, 0 0 0 8px ${v("cyan")}`,
          backgroundImage: GRID,
        }}
      >
        <p className="font-display" style={{ margin: 0, fontSize: 12, color: v("cyan"), letterSpacing: 2 }}>
          {card.name.toUpperCase()} SAYS
        </p>
        <h1
          className="font-display"
          style={{ margin: "24px 0 0", fontWeight: 400, fontSize: "clamp(22px, 6vw, 40px)", lineHeight: 1.35, color: v("yellow"), textShadow: `4px 4px 0 ${HEX_PINK}`, overflowWrap: "anywhere" }}
        >
          {headline(card)}
        </h1>
        <p className="font-display" style={{ margin: "20px 0 0", fontSize: 14, color: v("text") }}>
          SCORE {formatPts(card.pts)}
        </p>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 16, justifyContent: "center", marginTop: 32 }}>
          <Link href="/" className="font-display" style={{ ...cta, background: v("yellow"), color: "#2A0A3F", boxShadow: "0 6px 0 #B89A1A" }}>
            <svg width="14" height="14" viewBox="0 0 10 10" aria-hidden="true" focusable="false"><path fill="currentColor" d="M1 0l9 5-9 5z" /></svg>
            PLAY BREACHLAB
          </Link>
          <Link href="/daily" className="font-display" style={{ ...cta, border: `3px solid ${v("cyan")}`, color: v("cyan") }}>
            DAILY VAULT
          </Link>
        </div>
      </section>
      {/* eslint-disable-next-line @next/next/no-img-element -- the generated card is the point of this page */}
      <img
        src={`/api/og?${q}`}
        width={1200}
        height={630}
        alt={`Share card: ${headline(card)}. Score ${formatPts(card.pts)}. Player ${card.name}.`}
        style={{ width: "100%", height: "auto", borderRadius: 16, border: `2px solid ${v("border")}` }}
      />
      <ShareCard {...card} />
      <StatsTicker />
    </div>
  );
}

const HEX_PINK = "#FF3EA5";
// 48px arcade grid as a tiled SVG (hard 1px lines, no gradients).
const GRID = `url("data:image/svg+xml,${encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48"><path d="M0 .5H48M.5 0V48" stroke="#2A1259"/></svg>')}")`;
