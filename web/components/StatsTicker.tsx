"use client";
// Live stats ticker (CONTRACTS.md 7.8). Lane V. Lane U imports it as-is.
// Polls GET {NEXT_PUBLIC_API_BASE}/api/stats every 30 s; renders nothing until the first good answer; never throws.
import { useEffect, useState } from "react";
import { BOSSES, v } from "@/app/api/og/card";
import { API } from "@/lib/api";

export type Stats = {
  players_today: number; messages_today: number; bosses_defeated_today: number; blocks_today: number;
  last_breach: { boss: string; at: string } | null; updated_at: string;
};

export const POLL_MS = 30_000;

export function ago(at: string, now = Date.now()): string {
  const t = Date.parse(at);
  if (Number.isNaN(t)) return "";
  const min = Math.max(0, Math.floor((now - t) / 60_000));
  if (min < 1) return "JUST NOW";
  if (min < 60) return `${min} MIN AGO`;
  if (min < 1440) return `${Math.floor(min / 60)} H AGO`;
  return `${Math.floor(min / 1440)} D AGO`;
}

const n = (x: unknown) => (typeof x === "number" && Number.isFinite(x) ? Math.max(0, Math.floor(x)) : 0);

export function tickerText(s: Stats, now = Date.now()): string {
  const parts = [
    `${n(s.players_today)} ${n(s.players_today) === 1 ? "PLAYER" : "PLAYERS"} TODAY`,
    `${n(s.messages_today)} ${n(s.messages_today) === 1 ? "MESSAGE" : "MESSAGES"}`,
    `${n(s.bosses_defeated_today)} ${n(s.bosses_defeated_today) === 1 ? "BOSS" : "BOSSES"} DEFEATED`,
  ];
  const boss = s.last_breach?.boss;
  if (boss && Object.values(BOSSES).includes(boss)) parts.push(`LAST BREACH: ${boss.toUpperCase()} ${ago(s.last_breach!.at, now)}`.trim());
  return parts.join(" · ");
}

export default function StatsTicker() {
  const [stats, setStats] = useState<Stats | null>(null);
  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const r = await fetch(`${API}/api/stats`, { cache: "no-store" });
        if (!r.ok) return;
        const s = (await r.json()) as Stats;
        if (alive && s && typeof s === "object") setStats(s);
      } catch { /* fails silently: the ticker is decoration */ }
    };
    void load();
    const id = setInterval(load, POLL_MS);
    return () => { alive = false; clearInterval(id); };
  }, []);
  if (!stats) return null;
  const text = tickerText(stats);
  return (
    <div
      data-testid="stats-ticker"
      role="region"
      aria-label="Live stats"
      className="font-display"
      style={{
        overflow: "hidden", whiteSpace: "nowrap", borderRadius: 12, border: `2px solid ${v("border")}`,
        background: v("surface"), color: v("yellow"), fontSize: 11, lineHeight: "36px", minHeight: 36,
      }}
    >
      <style>{`
        .v-ticker { display: inline-flex; gap: 64px; padding-left: 16px; animation: v-ticker 30s linear infinite; }
        @keyframes v-ticker { from { transform: translateX(0); } to { transform: translateX(-50%); } }
        @media (prefers-reduced-motion: reduce) { .v-ticker { animation: none; } .v-ticker-dup { display: none; } }
      `}</style>
      <div className="v-ticker">
        <span>{text}</span>
        <span className="v-ticker-dup" aria-hidden="true">{text}</span>
      </div>
    </div>
  );
}
