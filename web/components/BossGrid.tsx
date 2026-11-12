"use client";
import Link from "next/link";
import { Check, Lock, Skip, Star } from "@/components/Icons";
import type { LevelState } from "@/lib/api";
import { prewarm } from "@/lib/api";
import { BOSSES, bossLabel } from "@/lib/game";
import { GUARD_LEVELS } from "@/lib/guard";

const starsFor = (used: number) => (used <= 3 ? 3 : used <= 8 ? 2 : 1);

/** SELECT BOSS grid. `levels` = normalised session state (null before a session exists: boss 1 open, rest locked). */
export default function BossGrid({ levels, current, unlockAll = false, hrefFor }: {
  levels: LevelState[] | null; current?: number; unlockAll?: boolean; hrefFor: (id: number) => string;
}) {
  return (
    <ul className="grid grid-cols-1 gap-4 min-[480px]:grid-cols-2 lg:grid-cols-5" data-testid="boss-grid">
      {BOSSES.map((b, i) => {
        const st = levels?.[i]?.status ?? (b.id === 1 ? "open" : "locked");
        const locked = st === "locked" && !unlockAll;
        const now = current === b.id;
        const progress = `${(b.id / 5) * 100}%`;
        const tone = now
          ? "bg-pink text-ink border-pink -rotate-[1.5deg] motion-reduce:rotate-0"
          : st === "cleared" ? "bg-surface border-green" : st === "skipped" ? "bg-surface border-muted-2" : locked ? "bg-surface border-dashed border-border text-muted-2" : "bg-surface border-cyan";
        const label = now ? "NOW PLAYING" : st === "cleared" ? "CLEARED" : st === "skipped" ? "SKIPPED" : locked ? (b.id === 5 ? "FINAL BOSS" : "LOCKED") : "READY";
        const labelTone = now ? "" : st === "cleared" ? "text-green" : st === "skipped" ? "text-muted" : locked ? "" : "text-cyan";
        const inner = (
          <>
            <span className={`flex items-center gap-2 font-display text-[10px] leading-4 ${labelTone}`}>
              {st === "cleared" && !now && <Check size={12} />}
              {st === "skipped" && !now && <Skip size={12} />}
              {locked && <Lock size={12} />}
              {label}
              {st === "cleared" && !now && (
                <span className="inline-flex text-yellow" aria-label={`${starsFor(levels?.[i]?.messages_used ?? 20)} of 3 stars`} role="img">
                  {Array.from({ length: starsFor(levels?.[i]?.messages_used ?? 20) }, (_, k) => <Star key={k} size={11} />)}
                </span>
              )}
            </span>
            <span className="mt-4 block text-xl font-bold leading-tight">{bossLabel(b.id)}</span>
            <span className={`mt-2 block text-[15px] ${now ? "" : locked ? "" : "text-muted"}`}>{b.tag}</span>
            <span className={`mt-4 block h-2.5 rounded-full ${now ? "bg-pink-dark" : "bg-raised"}`} aria-hidden>
              <span className={`block h-2.5 rounded-full ${now ? "bg-yellow" : st === "cleared" ? "bg-green" : locked ? "bg-border" : st === "skipped" ? "bg-muted-2" : "bg-cyan"}`} style={{ width: progress }} />
            </span>
          </>
        );
        const cls = `block h-full rounded-[20px] border-[3px] p-5 no-underline ${tone}`;
        return (
          <li key={b.id} data-testid={`boss-${b.id}`} data-status={st}
            onMouseEnter={GUARD_LEVELS.includes(b.id) ? prewarm : undefined} onFocus={GUARD_LEVELS.includes(b.id) ? prewarm : undefined}>
            {locked
              ? <div className={cls} aria-disabled="true">{inner}</div>
              : <Link href={hrefFor(b.id)} className={`${cls} text-text hover:text-text hover:brightness-110 ${now ? "text-ink hover:text-ink" : ""}`} aria-current={now ? "true" : undefined}>{inner}</Link>}
          </li>
        );
      })}
    </ul>
  );
}
