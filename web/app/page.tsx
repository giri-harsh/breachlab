"use client";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import BossGrid from "@/components/BossGrid";
import GuardBanner from "@/components/GuardBanner";
import HighScores from "@/components/HighScores";
import { Arrow, PlayIcon } from "@/components/Icons";
import StartForm from "@/components/StartForm";
import StatsTicker from "@/components/StatsTicker";
import { ApiError, call, warm, type Board, type LevelState, type SessionState } from "@/lib/api";
import { normaliseSession, store, withUnlocks, type SavedSession } from "@/lib/game";
import { useGuardState } from "@/lib/guard";

export default function Home() {
  const router = useRouter();
  const [saved, setSaved] = useState<SavedSession | null>(null);
  const [levels, setLevels] = useState<LevelState[] | null>(null);
  const [board, setBoard] = useState<Board | null>(null);
  const [boardErr, setBoardErr] = useState(false);
  const guardDown = useGuardState() === "offline";

  // Wake the guard while the player reads the page / types a name (ADR-0085); never blocks anything.
  useEffect(() => { warm(); }, []);
  useEffect(() => {
    call<Board>("/api/leaderboard").then(setBoard).catch(() => setBoardErr(true));
    const s = store.session();
    if (!s) return;
    setSaved(s);
    const cached = store.progress(s.id);
    if (cached) setLevels(cached.levels);
    call<SessionState>(`/api/session/${encodeURIComponent(s.id)}`)
      .then((st) => setLevels(withUnlocks(normaliseSession(st))))
      .catch((e) => { if (e instanceof ApiError && e.status === 404) { store.clearSession(); setSaved(null); setLevels(null); } });
  }, []);

  const nextBoss = levels?.find((l) => l.status === "open")?.id ?? 1;
  return (
    <div className="space-y-14 pb-6">
      <section className="cabinet grid-bg mt-2 px-5 py-12 text-center sm:px-10 sm:py-16" aria-labelledby="hero-title">
        <p className="anim-blink font-display text-[10px] tracking-[2px] text-cyan sm:text-[13px]">INSERT PROMPT TO CONTINUE</p>
        <h1 id="hero-title" className="mt-7 font-display text-[34px] leading-[1.3] text-yellow [text-shadow:4px_4px_0_var(--pink)] sm:text-[64px] sm:leading-[1.25] sm:[text-shadow:6px_6px_0_var(--pink)]">
          BREAK<br />THE BOT
        </h1>
        <p className="mx-auto mt-7 max-w-[560px] text-[17px] leading-relaxed text-soft sm:text-[19px]">
          Five bosses. One secret code each. Trick the AI into spilling it before you run out of messages.
        </p>
        <div className="mx-auto mt-9 max-w-[640px]">
          {guardDown && <GuardBanner />}
          {saved ? (
            <div className="flex flex-col items-center gap-4">
              <Link href={`/play?boss=${nextBoss}`} className="btn btn-yellow min-h-[60px] px-8 font-display text-sm text-ink no-underline hover:text-ink">
                <PlayIcon size={14} />CONTINUE AS {saved.nickname.toUpperCase()}
              </Link>
              <button type="button" onClick={() => { store.clearSession(); setSaved(null); setLevels(null); }} className="min-h-11 text-sm text-muted underline hover:text-text">
                New player
              </button>
            </div>
          ) : (
            <StartForm onStarted={() => router.push("/play?boss=1")} />
          )}
        </div>
      </section>

      <section aria-labelledby="select-boss">
        <h2 id="select-boss" className="mb-6 font-display text-base text-cyan sm:text-lg">SELECT BOSS</h2>
        <BossGrid levels={levels} hrefFor={(id) => `/play?boss=${id}`} />
      </section>

      <section aria-labelledby="hs" className="space-y-5">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <h2 id="hs" className="font-display text-base text-yellow sm:text-lg">HIGH SCORES</h2>
          <Link href="/leaderboard" className="inline-flex min-h-11 items-center gap-2 font-bold">All high scores <Arrow size={14} /></Link>
        </div>
        {boardErr ? <p className="text-muted">High scores unavailable right now.</p> : board ? <HighScores rows={board.leaderboard} limit={5} /> : <p className="text-muted">Loading high scores…</p>}
      </section>

      <StatsTicker />
    </div>
  );
}
