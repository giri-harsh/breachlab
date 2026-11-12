"use client";
import { useEffect, useState } from "react";
import HighScores from "@/components/HighScores";
import { Trophy } from "@/components/Icons";
import { call, type Board } from "@/lib/api";
import { bossLabel } from "@/lib/game";

export default function Leaderboard() {
  const [data, setData] = useState<Board | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => { call<Board>("/api/leaderboard").then(setData).catch(() => setErr("High scores unavailable right now.")); }, []);
  return (
    <div className="space-y-10">
      <section className="space-y-5">
        <h1 className="font-display text-xl leading-9 text-yellow sm:text-3xl">HIGH SCORES</h1>
        <p className="max-w-2xl text-soft">Ranked by bosses beaten, then points, then fewest messages. A grey <span className="chip bg-chip-grey-bg text-chip-grey">SKIP</span> chip marks a skipped boss (0 points).</p>
        {err && <p className="text-yellow">{err}</p>}
        {data ? <HighScores rows={data.leaderboard} /> : !err && <p className="text-muted">Loading…</p>}
      </section>
      <section className="space-y-3" aria-labelledby="hob">
        <h2 id="hob" className="flex items-center gap-2 font-display text-sm text-pink sm:text-base"><Trophy size={16} />HALL OF BREACHES</h2>
        <p className="text-sm text-muted">Recent wins against bosses 4 and 5. Prompts stay hidden.</p>
        <ul className="space-y-2">
          {data?.hall_of_breaches.map((h, i) => (
            <li key={i} className="rounded-[16px] border-[3px] border-border bg-surface px-4 py-3">
              <span className="font-bold">{h.nickname}</span> beat <span className="text-cyan">{bossLabel(h.level)}</span>
              <span className="text-muted"> ({h.guard_version ?? "unknown guard"})</span>
            </li>
          ))}
          {data && data.hall_of_breaches.length === 0 && <li className="text-muted">Nobody has broken the trained guard yet.</li>}
        </ul>
      </section>
    </div>
  );
}
