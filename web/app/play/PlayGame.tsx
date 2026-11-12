"use client";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { ErrorLine, Offline } from "@/components/Alerts";
import GuardBanner from "@/components/GuardBanner";
import Hearts from "@/components/Hearts";
import InsertName from "@/components/InsertName";
import { Arrow, Bolt, Coin, Lock, Skip, Skull, Trophy } from "@/components/Icons";
import { AssembleText, Confetti, Sheet, VerdictPill } from "@/components/Moments";
import ShareCard from "@/components/ShareCard";
import StartForm from "@/components/StartForm";
import {
  ApiError, call, prewarm, warm, type HintOut, type LevelState, type Reply, type SessionState, type SkipOut, type Started, type Verdict,
} from "@/lib/api";
import {
  BOSSES, DEFAULT_HINT_COST, DEFAULT_MAX_MESSAGES, MAX_CHARS, POWER_UPS, SKIP_CONFIRM, bossLabel, bumpHi, fmt, normaliseSession,
  store, verdictOf, withUnlocks, type SavedSession,
} from "@/lib/game";
import { GUARD_LEVELS, useGuardStatus } from "@/lib/guard";
import { blip } from "@/lib/sound";

type Msg = { id: number; who: "you" | "bot"; text: string; verdict?: Verdict; leaked?: boolean };
type Moment =
  | { kind: "win"; boss: number; msgs: number; pts: number }
  | { kind: "over"; boss: number }
  | { kind: "skipped"; boss: number; data: SkipOut };

let seq = 0;

export default function PlayGame() {
  const router = useRouter();
  const params = useSearchParams();
  const [ready, setReady] = useState(false);
  const [saved, setSaved] = useState<SavedSession | null>(null);
  const [levels, setLevels] = useState<LevelState[]>(() => withUnlocks(normaliseSession(null)));
  const [maxMsgs, setMaxMsgs] = useState(DEFAULT_MAX_MESSAGES);
  const [total, setTotal] = useState<number | null>(null);
  const [streak, setStreak] = useState(0);
  const [hintCost, setHintCost] = useState(DEFAULT_HINT_COST);
  const [logs, setLogs] = useState<Record<number, Msg[]>>({});
  const [left, setLeft] = useState<Record<number, number>>({});
  const [hints, setHints] = useState<Record<number, Record<number, string>>>({});
  const [askHint, setAskHint] = useState<number | null>(null);
  const [hintErr, setHintErr] = useState("");
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [waking, setWaking] = useState(false);
  const [error, setError] = useState("");
  const [offline, setOffline] = useState<"" | "guard" | "vault">("");
  const [guard503, setGuard503] = useState(false);
  const [moment, setMoment] = useState<Moment | null>(null);
  const [confetti, setConfetti] = useState(false);
  const [confirmSkip, setConfirmSkip] = useState(false);
  const [unlocked, setUnlocked] = useState(false);
  const gs = useGuardStatus();
  const guardDown = gs.state === "offline" || guard503;
  const input = useRef<HTMLTextAreaElement>(null);
  const end = useRef<HTMLDivElement>(null);

  const reqBoss = Number(params.get("boss"));
  const firstOpen = levels.find((l) => l.status === "open")?.id ?? 1;
  // Without ?boss the first open boss is played; `sticky` pins it once a message is sent so a clear does not jump ahead.
  const [sticky, setSticky] = useState<number | null>(null);
  const boss = reqBoss >= 1 && reqBoss <= 5 ? reqBoss : (sticky ?? firstOpen);
  const info = BOSSES[boss - 1];
  const lvl = levels[boss - 1];
  const msgs = logs[boss] ?? [];
  const msgsLeft = left[boss] ?? Math.max(0, maxMsgs - (lvl?.messages_used ?? 0));
  const locked = lvl?.status === "locked" && !unlocked;
  const cleared = lvl?.status === "cleared";
  // ?unlock=1 (tests) keeps the composer usable so the live spec can observe the real 503 from the gateway.
  const levelOffline = guardDown && GUARD_LEVELS.includes(boss) && !unlocked;
  // A cold guard is not an error: bosses 3-5 show BOSS WAKING UP… and the composer opens by itself when it answers (or after 30 s).
  const guardWaking = gs.state === "warming" && !gs.expired && !guard503 && GUARD_LEVELS.includes(boss) && !unlocked;
  const out = msgsLeft <= 0 && !cleared;
  const blockedComposer = locked || levelOffline || guardWaking || cleared || out;
  const pendingWin = useRef<Moment | null>(null);

  const applyState = useCallback((st: SessionState) => {
    const ls = withUnlocks(normaliseSession(st));
    setLevels(ls);
    if (typeof st.max_messages === "number") setMaxMsgs(st.max_messages);
    if (typeof st.total_points === "number") { setTotal(st.total_points); bumpHi(st.total_points); }
    if (typeof st.streak === "number") setStreak(st.streak);
    if (typeof st.hint_cost === "number") setHintCost(st.hint_cost);
    return ls;
  }, []);

  const loadSession = useCallback(async (s: SavedSession) => {
    try {
      const st = await call<SessionState>(`/api/session/${encodeURIComponent(s.id)}`);
      const ls = applyState(st);
      store.setProgress(s.id, ls, st.total_points ?? 0);
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) { store.clearSession(); setSaved(null); }
    }
  }, [applyState]);

  // Test-only: ?unlock=1 enables every boss in the UI. The server still enforces gating (403) for real sessions.
  useEffect(() => { setUnlocked(params.get("unlock") === "1"); }, [params]);
  useEffect(() => {
    warm();
    const s = store.session();
    if (s) {
      setSaved(s);
      const cached = store.progress(s.id);
      if (cached) setLevels(withUnlocks(cached.levels));
      void loadSession(s);
    }
    setReady(true);
  }, [loadSession]);
  useEffect(() => { end.current?.scrollIntoView({ behavior: "smooth", block: "nearest" }); }, [msgs.length, waking]);
  useEffect(() => { setError(""); setOffline(""); setAskHint(null); setHintErr(""); input.current?.focus(); }, [boss]);
  useEffect(() => { if (!guardWaking) input.current?.focus(); }, [guardWaking]);

  const persist = (ls: LevelState[], t: number | null) => { if (saved) store.setProgress(saved.id, ls, t ?? 0); };
  const push = (level: number, m: Omit<Msg, "id">) => setLogs((l) => ({ ...l, [level]: [...(l[level] ?? []), { ...m, id: ++seq }] }));

  function started(s: SavedSession) {
    setSaved(s); setLogs({}); setLeft({}); setHints({}); setTotal(null); setStreak(0); setMoment(null); setSticky(null);
    setLevels(withUnlocks(normaliseSession(null)));
    void loadSession(s);
  }

  // 503 (after the client's one retry) means the guard or the vault is down: say so instead of a bare error.
  function fail(err: unknown, fallback: string) {
    if (err instanceof ApiError && err.status === 503) {
      const kind = /vault/i.test(err.message) ? "vault" : "guard";
      if (kind === "guard") setGuard503(true);
      setOffline(kind); setError(""); return;
    }
    if (err instanceof ApiError && err.status === 409 && /cap|limit|messages/i.test(err.message)) { setMoment({ kind: "over", boss }); return; }
    setOffline(""); setError(err instanceof ApiError ? err.message : fallback);
  }

  async function send() {
    const msg = text.trim();
    if (!saved || !msg || busy || blockedComposer) return;
    const level = boss;
    setSticky(level);
    setText(""); setError(""); setOffline(""); setBusy(true); blip("fire");
    push(level, { who: "you", text: msg });
    try {
      const r = await call<Reply>("/api/message", { method: "POST", body: JSON.stringify({ session_id: saved.id, level, message: msg }) }, undefined, () => setWaking(true));
      const v = verdictOf(r);
      const reply = r.blocked
        ? (v.layer === "output_filter" ? "The output filter scrubbed my reply. Nothing leaves this vault." : `Guard blocked this message before it reached the vault (${v.category}).`)
        : (r.reply || "(no reply)");
      // the win moment (confetti + card) starts when the leaked reply has finished assembling: see onAssembled
      if (r.level_cleared) pendingWin.current = { kind: "win", boss: level, msgs: maxMsgs - r.messages_left, pts: r.points_awarded ?? 0 };
      push(level, { who: "bot", text: reply, verdict: v, leaked: r.level_cleared });
      setLeft((m) => ({ ...m, [level]: r.messages_left }));
      if (typeof r.total_points === "number") { setTotal(r.total_points); bumpHi(r.total_points); }
      if (typeof r.streak === "number") setStreak(r.streak);
      const next = levels.map((l) => (l.id === level ? { ...l, messages_used: maxMsgs - r.messages_left, status: r.level_cleared ? "cleared" as const : l.status, points: r.level_cleared ? (r.points_awarded ?? l.points) : l.points } : l));
      const ls = withUnlocks(next);
      setLevels(ls); persist(ls, r.total_points ?? total);
      if (!r.level_cleared) {
        blip(r.blocked ? "block" : "ok");
        if (r.messages_left <= 0) { blip("lose"); setMoment({ kind: "over", boss: level }); }
      }
    } catch (err) { fail(err, "Network error."); }
    finally { setBusy(false); setWaking(false); input.current?.focus(); }
  }

  const onAssembled = useCallback(() => {
    const w = pendingWin.current;
    if (!w) return;
    pendingWin.current = null;
    blip("win"); setConfetti(true); setMoment(w);
  }, []);

  async function skip() {
    if (!saved) return;
    const level = boss;
    setConfirmSkip(false); setError("");
    try {
      const r = await call<SkipOut>(`/api/session/${encodeURIComponent(saved.id)}/skip`, { method: "POST", body: JSON.stringify({ level }) });
      const ls = withUnlocks(levels.map((l) => (l.id === level ? { ...l, status: "skipped" as const, points: 0 } : l)));
      setLevels(ls); setTotal(r.total_points); setStreak(0); persist(ls, r.total_points);
      setMoment({ kind: "skipped", boss: level, data: r });
    } catch (err) { setMoment(null); fail(err, "Could not skip right now."); }
  }

  async function newRun() {
    if (!saved) return;
    try {
      const s = await call<Started>("/api/session", { method: "POST", body: JSON.stringify({ nickname: saved.nickname }) });
      const next = { id: s.session_id, nickname: saved.nickname, consent: s.consent };
      store.setSession(next); warm();
      started(next);
      router.replace("/play?boss=1");
    } catch (err) { setMoment(null); fail(err, "Could not start a new run."); }
  }

  async function reveal(tier: number) {
    if (!saved) return;
    setHintErr("");
    try {
      const r = await call<HintOut>(`/api/session/${encodeURIComponent(saved.id)}/hint`, { method: "POST", body: JSON.stringify({ level: boss, tier }) });
      setHints((h) => ({ ...h, [boss]: { ...(h[boss] ?? {}), [tier]: r.text } }));
      if (typeof r.total_points === "number") setTotal(r.total_points);
      setStreak(0);
      const used = Array.isArray(r.hints_used) ? r.hints_used : Array.from({ length: Number(r.hints_used) || tier }, (_, i) => i + 1);
      setLevels((ls) => ls.map((l) => (l.id === boss ? { ...l, hints_used: used } : l)));
      setAskHint(null);
    } catch (err) {
      setHintErr(err instanceof ApiError ? (err.status === 429 ? "Too many hint requests. Wait a minute." : err.message) : "Hint unavailable right now.");
    }
  }

  function powerUp(starter: string) {
    setText(starter.slice(0, MAX_CHARS));
    requestAnimationFrame(() => { const el = input.current; if (el) { el.focus(); el.setSelectionRange(el.value.length, el.value.length); } });
  }

  function onKey(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); void send(); }
    else if (e.key === "Escape") { e.preventDefault(); setText(""); }
  }

  if (!ready) return <InsertName />;
  if (!saved) return <InsertName>{guardDown && <GuardBanner />}<StartForm onStarted={started} autoFocus /></InsertName>;

  const revealed = hints[boss] ?? {};
  const used = lvl?.hints_used ?? [];
  const isRevealed = (t: number) => t in revealed || used.includes(t);

  return (
    <div className="pb-4">
      {guardDown && <GuardBanner />}
      <nav aria-label="Bosses" className="-mx-1 mb-5 flex gap-2 overflow-x-auto px-1 pb-2">
        {levels.map((l) => {
          const off = l.status === "locked" && !unlocked;
          const cur = l.id === boss;
          const tone = cur ? "border-pink bg-pink text-ink" : l.status === "cleared" ? "border-green text-green" : l.status === "skipped" ? "border-muted-2 text-muted" : off ? "border-border text-muted-2" : "border-cyan text-cyan";
          return (
            <button key={l.id} type="button" disabled={off} aria-current={cur ? "true" : undefined} data-testid={`boss-tab-${l.id}`} data-status={l.status}
              aria-label={`Boss ${bossLabel(l.id)}${l.status === "cleared" ? ", cleared" : l.status === "skipped" ? ", skipped" : off ? ", locked" : ""}${guardDown && GUARD_LEVELS.includes(l.id) ? ", offline" : gs.state === "warming" && !gs.expired && GUARD_LEVELS.includes(l.id) ? ", waking up" : ""}`}
              onClick={() => router.push(`/play?boss=${l.id}${unlocked ? "&unlock=1" : ""}`)}
              onMouseEnter={GUARD_LEVELS.includes(l.id) ? prewarm : undefined} onFocus={GUARD_LEVELS.includes(l.id) ? prewarm : undefined}
              className={`inline-flex min-h-11 shrink-0 items-center gap-1.5 rounded-[12px] border-[3px] px-3 font-display text-[10px] disabled:cursor-not-allowed ${tone}`}>
              {off ? <Lock size={10} /> : l.status === "skipped" ? <Skip size={10} /> : null}
              {l.id}<span className="hidden sm:inline">&nbsp;{BOSSES[l.id - 1].name.toUpperCase()}</span>
              {l.status === "skipped" && <span className="sr-only sm:not-sr-only"> SKIP</span>}
            </button>
          );
        })}
      </nav>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_380px]">
        <section aria-labelledby="boss-title" className="rounded-[24px] border-[3px] border-cyan bg-surface p-4 sm:p-6">
          <header className="flex flex-wrap items-center justify-between gap-3">
            <h1 id="boss-title" className="font-display text-[11px] leading-5 text-cyan" data-testid="boss-title">BOSS {bossLabel(boss).toUpperCase()}</h1>
            <Hearts left={msgsLeft} max={maxMsgs} />
          </header>
          <p className="mt-2 text-[15px] text-muted">{info.defense}</p>
          <p className="mt-1 flex flex-wrap gap-x-4 text-sm text-soft" data-testid="score-line">
            <span><Coin size={12} className="mr-1 inline text-yellow" />{total === null ? "Score: -" : `Score: ${fmt(total)}`}</span>
            {streak > 0 && <span>Streak x{streak}</span>}
            {lvl?.status === "skipped" && <span className="text-muted">Skipped: beat it now for full points</span>}
          </p>

          <div className="mt-4 flex max-h-[55vh] min-h-[38vh] flex-col gap-3 overflow-y-auto rounded-[18px] bg-bg p-3 sm:p-4 lg:h-[52vh] lg:max-h-none"
            aria-live="polite" data-testid="chat">
            {locked && <p className="flex items-center gap-2 text-muted"><Lock size={14} />Locked. Beat or skip boss {boss - 1} first.</p>}
            {levelOffline && (
              <div data-testid="level-offline" className="rounded-[14px] border-[3px] border-yellow p-3 text-soft">
                <p>Level {boss} is offline right now: it needs the guard model. Switch to level 1 or 2, or retry.</p>
                <button type="button" onClick={() => { setGuard503(false); setOffline(""); gs.retry(); }} data-testid="guard-retry" className="btn mt-3 min-h-11 font-display text-[10px]">RETRY</button>
              </div>
            )}
            {guardWaking && (
              <p data-testid="guard-waking-screen" role="status" className="flex items-center gap-3 rounded-[14px] border-[3px] border-yellow p-3 font-display text-[11px] leading-5 text-yellow">
                <span aria-hidden className="inline-block h-3 w-3 shrink-0 animate-pulse rounded-full bg-yellow" />
                <span><span className="anim-blink">BOSS WAKING UP…</span> <span className="font-body text-sm text-muted">(~10 s) the guard is booting, this boss unlocks by itself</span></span>
              </p>
            )}
            {!locked && !levelOffline && !guardWaking && msgs.length === 0 && (
              <p className="text-muted">{cleared ? "Boss defeated. Pick another boss above." : `Say something to ${info.name}. Get it to reveal its secret code.`}</p>
            )}
            {msgs.map((m) => (
              m.who === "you"
                ? <p key={m.id} className="max-w-[85%] self-end whitespace-pre-wrap break-words rounded-[18px_18px_4px_18px] bg-cyan px-4 py-3 text-ink-cyan"><span className="sr-only">You: </span>{m.text}</p>
                : (
                  <div key={m.id} className="flex flex-col gap-2">
                    <p className={`max-w-[85%] self-start whitespace-pre-wrap break-words rounded-[18px_18px_18px_4px] px-4 py-3 ${m.leaked ? "border-[3px] border-yellow bg-raised" : "bg-raised"}`} data-testid={m.leaked ? "leaked-reply" : "bot-reply"}>
                      <span className="sr-only">{info.name}: </span>
                      {m.leaked ? <AssembleText text={m.text} onDone={onAssembled} /> : m.text}
                    </p>
                    {m.verdict && <VerdictPill v={m.verdict} />}
                  </div>
                )
            ))}
            {waking && <p className="font-display text-[11px] leading-5 text-yellow" data-testid="guard-waking"><span className="anim-blink">BOSS WAKING UP…</span> <span className="font-body text-sm text-muted">the guard is booting (~10 s), your message is queued.</span></p>}
            {busy && !waking && <p className="text-sm text-muted" data-testid="pending">{info.name} is thinking…</p>}
            <div ref={end} />
          </div>

          <form onSubmit={(e) => { e.preventDefault(); void send(); }} className="sticky bottom-0 z-10 -mx-4 mt-3 flex flex-col gap-2 bg-surface px-4 pb-3 pt-2 sm:-mx-6 sm:px-6 lg:static lg:mx-0 lg:px-0 lg:pb-0">
            <p className="text-xs text-muted" data-testid="consent">{saved.consent}</p>
            <label htmlFor="msg" className="sr-only">Your message</label>
            <div className="flex items-end gap-2">
              <textarea id="msg" ref={input} value={text} onChange={(e) => setText(e.target.value.slice(0, MAX_CHARS))} onKeyDown={onKey}
                rows={2} autoFocus disabled={blockedComposer} placeholder="Type your move… (Enter to fire)" maxLength={MAX_CHARS}
                aria-describedby="msg-help"
                className="field min-h-[52px] flex-1 resize-none" />
              <button disabled={blockedComposer || busy || !text.trim()} className="btn btn-yellow min-h-[52px] shrink-0 font-display text-xs">FIRE</button>
            </div>
            <div className="flex items-center justify-between gap-3 text-xs text-muted" id="msg-help">
              <span>Enter fires · Shift+Enter new line · Esc clears</span>
              <span data-testid="char-count" className={text.length >= MAX_CHARS ? "text-yellow" : ""}>{text.length}/{MAX_CHARS}</span>
            </div>
            {error && <ErrorLine text={error} />}
            {offline && <Offline kind={offline} />}
          </form>
        </section>

        <aside className="flex flex-col gap-5" aria-label="Power-ups, hints and skip">
          <section aria-labelledby="pu" className="rounded-[24px] bg-yellow p-5 text-ink">
            <h2 id="pu" className="flex items-center gap-2 font-display text-[12px]"><Bolt size={14} />POWER-UPS</h2>
            <p className="mt-2 text-sm">Tap one to load a starter prompt. Edit it, then FIRE.</p>
            <div className="mt-4 grid grid-cols-2 gap-2.5">
              {POWER_UPS.map((p) => (
                <button key={p.id} type="button" onClick={() => powerUp(p.starter)} disabled={blockedComposer} data-testid={`powerup-${p.id}`}
                  className="min-h-12 rounded-[12px] border-2 border-ink px-2 py-1.5 text-left font-bold leading-tight hover:bg-ink hover:text-yellow disabled:opacity-50">
                  {p.name}<span className="block text-xs font-medium">{p.blurb}</span>
                </button>
              ))}
            </div>
          </section>

          <section aria-labelledby="ht" className="rounded-[24px] border-[3px] border-pink bg-surface p-5">
            <h2 id="ht" className="font-display text-[12px] text-pink">HINT TOKENS</h2>
            <p className="mt-2 text-sm text-soft">Stuck? Spend a token. Each one costs {hintCost} pts and resets your streak.</p>
            <div className="mt-3 flex flex-wrap gap-2.5">
              {[1, 2, 3].map((t) => {
                const can = t === 1 || isRevealed(t - 1);
                const done = isRevealed(t);
                return (
                  <button key={t} type="button" disabled={!can || locked} data-testid={`hint-${t}`}
                    onClick={() => (done ? void reveal(t) : setAskHint(t))}
                    className={`min-h-11 rounded-[12px] border-2 px-4 font-bold ${done ? "border-green text-green" : can ? "border-pink text-pink" : "border-border text-muted-2"} disabled:cursor-not-allowed`}>
                    Hint {t}{done ? "" : ` · ${hintCost} pts`}
                  </button>
                );
              })}
            </div>
            {askHint !== null && (
              <div className="mt-3 rounded-[14px] border-2 border-pink p-3" data-testid="hint-confirm">
                <p className="text-sm">Hint {askHint} costs <strong className="text-yellow">{hintCost} pts</strong>. Reveal it?</p>
                <div className="mt-2 flex gap-2">
                  <button type="button" onClick={() => void reveal(askHint)} className="btn btn-pink min-h-11 text-sm">Reveal (-{hintCost})</button>
                  <button type="button" onClick={() => setAskHint(null)} className="btn btn-ghost min-h-11 text-sm">Cancel</button>
                </div>
              </div>
            )}
            {hintErr && <p role="alert" className="mt-2 text-sm text-yellow">{hintErr}</p>}
            {Object.keys(revealed).length > 0 && (
              <ol className="mt-3 space-y-2 text-sm" data-testid="hint-list">
                {Object.entries(revealed).sort(([a], [b]) => Number(a) - Number(b)).map(([t, h]) => (
                  <li key={t} className="rounded-[12px] bg-raised p-3"><span className="font-bold text-pink">Hint {t}: </span>{h}</li>
                ))}
              </ol>
            )}
          </section>

          {!cleared && !locked && (
            <button type="button" onClick={() => setConfirmSkip(true)} className="btn btn-ghost min-h-12 self-start" data-testid="skip-boss">
              <Skip size={14} />SKIP BOSS
            </button>
          )}
        </aside>
      </div>

      {confetti && <Confetti onEnd={() => setConfetti(false)} />}

      {confirmSkip && (
        <Sheet title="SKIP BOSS?" labelId="skip-title" tone="pink" onClose={() => setConfirmSkip(false)} testId="skip-confirm">
          <p className="text-soft">{SKIP_CONFIRM}</p>
          <div className="mt-5 flex flex-wrap gap-3">
            <button type="button" onClick={() => void skip()} className="btn btn-pink">Skip {info.name}</button>
            <button type="button" onClick={() => setConfirmSkip(false)} className="btn btn-ghost">Keep playing</button>
          </div>
        </Sheet>
      )}

      {moment?.kind === "win" && (
        <Sheet title={<span className="flex items-center gap-2"><Trophy size={16} />BOSS DEFEATED · {moment.msgs} msgs · +{fmt(moment.pts)}</span>} labelId="win-title" tone="green" onClose={() => setMoment(null)} testId="boss-defeated">
          <p className="text-soft">{bossLabel(moment.boss)} spilled the code. {moment.boss < 5 ? "The next boss is unlocked." : "You beat every boss."}</p>
          <div className="mt-4" data-testid="share-card-slot"><ShareCard boss={moment.boss} msgs={moment.msgs} pts={moment.pts} name={saved.nickname} /></div>
          <div className="mt-5 flex flex-wrap gap-3">
            {moment.boss < 5
              ? <button type="button" className="btn btn-green" onClick={() => { setMoment(null); router.push(`/play?boss=${moment.boss + 1}${unlocked ? "&unlock=1" : ""}`); }}>NEXT BOSS <Arrow size={14} /></button>
              : <Link href="/leaderboard" className="btn btn-green text-ink no-underline hover:text-ink">HIGH SCORES <Arrow size={14} /></Link>}
            <button type="button" className="btn btn-ghost" onClick={() => setMoment(null)}>Stay here</button>
          </div>
        </Sheet>
      )}

      {moment?.kind === "over" && (
        <Sheet title={<span className="flex items-center gap-2"><Skull size={16} />GAME OVER · CONTINUE?</span>} labelId="over-title" tone="pink" onClose={() => setMoment(null)} testId="game-over">
          <p className="text-soft">Out of messages for {bossLabel(moment.boss)}. Start a new run with the same name, or skip this boss and come back to it later.</p>
          <div className="mt-5 flex flex-wrap gap-3">
            <button type="button" className="btn btn-yellow" onClick={() => void newRun()}>CONTINUE (new run)</button>
            <button type="button" className="btn btn-ghost" onClick={() => { setMoment(null); setConfirmSkip(true); }}><Skip size={14} />SKIP BOSS</button>
          </div>
        </Sheet>
      )}

      {moment?.kind === "skipped" && (
        <Sheet title="HOW OTHERS BEAT IT" labelId="others-title" tone="cyan" onClose={() => setMoment(null)} testId="how-others">
          <p className="text-sm text-muted">{bossLabel(moment.boss)} is marked SKIPPED. You can come back anytime.</p>
          {moment.data.how_others_beat_it.length > 0 ? (
            <ul className="mt-3 space-y-3">
              {moment.data.how_others_beat_it.map((o, i) => (
                <li key={i} className="rounded-[14px] bg-raised p-3">
                  <p className="whitespace-pre-wrap break-words">{o.prompt}</p>
                  <p className="mt-1 text-xs text-muted">won in {o.messages} message{o.messages === 1 ? "" : "s"}</p>
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-3 rounded-[14px] bg-raised p-3" data-testid="fallback-hint">{moment.data.fallback_hint ?? "No winning prompts shared yet for this boss."}</p>
          )}
          <div className="mt-5 flex flex-wrap gap-3">
            {moment.data.next_level
              ? <button type="button" className="btn btn-cyan" onClick={() => { const n = moment.data.next_level; setMoment(null); router.push(`/play?boss=${n}${unlocked ? "&unlock=1" : ""}`); }}>NEXT BOSS <Arrow size={14} /></button>
              : <Link href="/leaderboard" className="btn btn-cyan text-ink-cyan no-underline hover:text-ink-cyan">HIGH SCORES <Arrow size={14} /></Link>}
            <button type="button" className="btn btn-ghost" onClick={() => setMoment(null)}>Close</button>
          </div>
        </Sheet>
      )}
    </div>
  );
}
