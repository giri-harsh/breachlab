"use client";
// Game moments: verdict pill, the leaked reply assembling, confetti, and the bottom sheet used by every overlay.
import { useEffect, useRef, useState } from "react";
import type { Verdict } from "@/lib/api";
import { prefersReducedMotion, verdictText } from "@/lib/game";

export function VerdictPill({ v }: { v: Verdict }) {
  const blocked = v.decision === "BLOCKED";
  return (
    <span data-testid="verdict" data-decision={v.decision}
      className={`pill self-center ${blocked ? "bg-pink text-ink" : "border-2 border-green bg-surface text-green"}`}>
      {verdictText(v)}
    </span>
  );
}

const GLYPHS = "ABCDEFGHJKMNPQRSTUVWXYZ23456789#$%&*+=?";
/** The leaked reply's characters assemble one by one (scrambled glyphs settle left to right), then onDone fires. */
export function AssembleText({ text, onDone }: { text: string; onDone?: () => void }) {
  const [shown, setShown] = useState(() => (prefersReducedMotion() ? text.length : 0));
  const [tick, setTick] = useState(0);
  const done = useRef(false);
  useEffect(() => {
    if (prefersReducedMotion()) { setShown(text.length); return; }
    const step = Math.max(8, Math.min(35, Math.floor(1400 / Math.max(1, text.length))));
    const id = setInterval(() => {
      setTick((t) => t + 1);
      setShown((n) => { if (n >= text.length) { clearInterval(id); return n; } return n + 1; });
    }, step);
    return () => clearInterval(id);
  }, [text]);
  useEffect(() => {
    if (shown >= text.length && !done.current) { done.current = true; onDone?.(); }
  }, [shown, text.length, onDone]);
  const tail = text.slice(shown, shown + 6).replace(/\S/g, (_, i: number) => GLYPHS[(i * 7 + tick * 13 + shown) % GLYPHS.length]);
  return (
    <>
      <span className="sr-only">{text}</span>
      <span aria-hidden data-testid="assemble" data-done={shown >= text.length}>
        {text.slice(0, shown)}<span className="text-pink">{shown < text.length ? tail : ""}</span>
      </span>
    </>
  );
}

const COLORS = ["#FF3EA5", "#3EF0FF", "#FFE14D", "#5BFF8A", "#F2EAFF"];
/** Canvas confetti, ~2.4 s, no library. Skipped entirely under prefers-reduced-motion. */
export function Confetti({ onEnd }: { onEnd?: () => void }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c || prefersReducedMotion()) { onEnd?.(); return; }
    const ctx = c.getContext("2d");
    if (!ctx) { onEnd?.(); return; }
    const w = (c.width = window.innerWidth), h = (c.height = window.innerHeight);
    const parts = Array.from({ length: 160 }, (_, i) => ({
      x: w / 2 + (Math.random() - 0.5) * w * 0.3, y: h * 0.35, vx: (Math.random() - 0.5) * 14, vy: -Math.random() * 14 - 4,
      s: 6 + Math.random() * 6, r: Math.random() * Math.PI, vr: (Math.random() - 0.5) * 0.3, c: COLORS[i % COLORS.length],
    }));
    let raf = 0; const t0 = performance.now();
    const frame = (t: number) => {
      ctx.clearRect(0, 0, w, h);
      for (const p of parts) {
        p.vy += 0.35; p.x += p.vx; p.y += p.vy; p.r += p.vr; p.vx *= 0.99;
        ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.r); ctx.fillStyle = p.c; ctx.fillRect(-p.s / 2, -p.s / 4, p.s, p.s / 2); ctx.restore();
      }
      if (t - t0 < 2400) raf = requestAnimationFrame(frame); else { ctx.clearRect(0, 0, w, h); onEnd?.(); }
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [onEnd]);
  return <canvas ref={ref} aria-hidden data-testid="confetti" className="pointer-events-none fixed inset-0 z-[60] h-full w-full" />;
}

/** Bottom sheet on phones, centred dialog on desktop. Focuses its first button; Esc closes when onClose is given. */
export function Sheet({ title, labelId, children, onClose, tone = "cyan", testId }: {
  title: React.ReactNode; labelId: string; children: React.ReactNode; onClose?: () => void; tone?: "cyan" | "pink" | "yellow" | "green"; testId?: string;
}) {
  const panel = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const prev = document.activeElement as HTMLElement | null;
    panel.current?.querySelector<HTMLElement>("button, a[href]")?.focus();
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape" && onClose) { e.preventDefault(); onClose(); } };
    window.addEventListener("keydown", onKey);
    return () => { window.removeEventListener("keydown", onKey); prev?.focus?.(); };
  }, [onClose]);
  const border = { cyan: "border-cyan", pink: "border-pink", yellow: "border-yellow", green: "border-green" }[tone];
  const head = { cyan: "text-cyan", pink: "text-pink", yellow: "text-yellow", green: "text-green" }[tone];
  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center sm:items-center" style={{ background: "rgba(18, 6, 43, 0.86)" }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose?.(); }}>
      <div ref={panel} role="dialog" aria-modal="true" aria-labelledby={labelId} data-testid={testId}
        className={`anim-pop max-h-[90vh] w-full overflow-y-auto rounded-t-[24px] border-[3px] ${border} bg-surface p-6 sm:max-w-lg sm:rounded-[24px]`}>
        <h2 id={labelId} className={`font-display text-[13px] leading-6 ${head}`}>{title}</h2>
        <div className="mt-4">{children}</div>
      </div>
    </div>
  );
}
