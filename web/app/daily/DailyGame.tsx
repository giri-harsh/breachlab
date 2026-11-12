"use client";
// Daily Vault game (CONTRACTS.md 7.7): GET /api/daily for the board, the normal session (POST /api/session),
// POST /api/message {level: 6}. 202 "warming" is retried inside `call` (ADR-0085). Lane V.
import { useCallback, useEffect, useRef, useState } from "react";
import ShareCard from "@/components/ShareCard";
import StatsTicker from "@/components/StatsTicker";
import { v } from "@/app/api/og/card";
import { ApiError, call, warm, type Reply, type Started } from "@/lib/api";

export type DailyBoard = { rank: number; nickname: string; messages: number; at: string };
export type Daily = { date: string; level: number; boss: string; max_messages: number; board: DailyBoard[] };
type Verdict = { decision: "BLOCKED" | "PASSED"; p_attack: number | null; category: string; layer: string };
type Reply11 = Reply & { verdict?: Verdict; points_awarded?: number; total_points?: number };
type Line = { who: "you" | "vault" | "guard"; text: string };

export const NICK_KEY = "breachlab.nickname";
export const SESSION_KEY = "breachlab.session";
const DAILY_LEVEL = 6;

function store(k: string, val?: string | null): string | null {
  try {
    if (val === undefined) return localStorage.getItem(k);
    if (val === null) localStorage.removeItem(k);
    else localStorage.setItem(k, val);
  } catch { /* storage blocked: play without remembering */ }
  return val ?? null;
}

const panel: React.CSSProperties = { borderRadius: 24, background: v("surface"), padding: 20 };
const input: React.CSSProperties = {
  flex: 1, minWidth: 0, minHeight: 52, borderRadius: 14, border: `2px solid ${v("border")}`, background: v("bg"),
  color: v("text"), padding: "0 16px", fontSize: 15,
};
const fire: React.CSSProperties = {
  minHeight: 52, minWidth: 88, padding: "0 20px", borderRadius: 14, border: 0, background: v("yellow"), color: "#2A0A3F",
  fontWeight: 700, cursor: "pointer",
};

function verdictText(vd: Verdict | undefined, fallback: string | null): string {
  if (!vd) return `BLOCKED${fallback ? ` · ${fallback}` : ""}`;
  const pct = vd.p_attack == null ? "" : ` · ${Math.round(vd.p_attack * 100)}% attack`;
  const cat = vd.category && vd.category !== "none" ? ` · ${vd.category}` : "";
  return `${vd.decision}${pct}${cat}`;
}

export default function DailyGame() {
  const [daily, setDaily] = useState<Daily | null>(null);
  const [boardErr, setBoardErr] = useState("");
  const [nick, setNick] = useState("");
  const [nickDraft, setNickDraft] = useState("");
  const [sid, setSid] = useState<string | null>(null);
  const [lines, setLines] = useState<Line[]>([]);
  const [msg, setMsg] = useState("");
  const [left, setLeft] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [waking, setWaking] = useState(false);
  const [err, setErr] = useState("");
  const [won, setWon] = useState<{ msgs: number; pts: number } | null>(null);
  const sidRef = useRef<string | null>(null);

  const loadBoard = useCallback(async () => {
    try {
      setDaily(await call<Daily>("/api/daily"));
      setBoardErr("");
    } catch {
      setBoardErr("Board unavailable right now.");
    }
  }, []);

  useEffect(() => {
    warm();
    void loadBoard();
    const n = store(NICK_KEY) ?? "";
    setNick(n);
    setNickDraft(n);
    sidRef.current = store(SESSION_KEY);
    setSid(sidRef.current);
  }, [loadBoard]);

  const max = daily?.max_messages ?? 10;
  const remaining = left ?? max;

  async function newSession(name: string): Promise<string> {
    const s = await call<Started>("/api/session", { method: "POST", body: JSON.stringify({ nickname: name }) });
    sidRef.current = s.session_id;
    setSid(s.session_id);
    store(SESSION_KEY, s.session_id);
    return s.session_id;
  }

  async function onStart(e: React.FormEvent) {
    e.preventDefault();
    const name = nickDraft.trim();
    if (name.length < 3 || name.length > 20) return setErr("Pick a name of 3-20 characters.");
    setErr("");
    setBusy(true);
    try {
      store(NICK_KEY, name);
      setNick(name);
      await newSession(name);
    } catch (x) {
      setErr(x instanceof ApiError ? x.message : "Could not start a session. Try again.");
    } finally {
      setBusy(false);
    }
  }

  async function send(text: string): Promise<Reply11> {
    const post = (id: string) =>
      call<Reply11>("/api/message", { method: "POST", body: JSON.stringify({ session_id: id, level: DAILY_LEVEL, message: text }) }, undefined, () => setWaking(true));
    const id = sidRef.current ?? (await newSession(nick));
    try {
      return await post(id);
    } catch (x) {
      if (x instanceof ApiError && x.status === 404) return post(await newSession(nick)); // stored session expired
      throw x;
    }
  }

  async function onFire(e: React.FormEvent) {
    e.preventDefault();
    const text = msg.trim();
    if (!text || busy || won || remaining <= 0) return;
    setBusy(true);
    setErr("");
    setLines((l) => [...l, { who: "you", text }]);
    setMsg("");
    try {
      const r = await send(text);
      setLeft(r.messages_left);
      setLines((l) => [...l, r.blocked ? { who: "guard", text: verdictText(r.verdict, r.block_category) } : { who: "vault", text: r.reply }]);
      if (r.level_cleared) {
        setWon({ msgs: Math.max(1, max - r.messages_left), pts: r.points_awarded ?? 0 });
        void loadBoard();
      }
    } catch (x) {
      setErr(x instanceof ApiError ? x.message : "Network error. Your message was not counted; try again.");
    } finally {
      setBusy(false);
      setWaking(false);
    }
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24, color: v("text") }}>
      <section style={{ border: `4px solid ${v("pink")}`, borderRadius: 28, padding: "28px 20px", textAlign: "center", boxShadow: `0 0 0 4px ${v("bg")}, 0 0 0 8px ${v("cyan")}` }}>
        <p className="font-display" style={{ margin: 0, fontSize: 12, color: v("cyan"), letterSpacing: 2 }}>
          BOSS 6 · {daily?.date ?? "TODAY"} UTC
        </p>
        <h1 className="font-display" style={{ margin: "20px 0 0", fontWeight: 400, fontSize: "clamp(26px, 7vw, 44px)", color: v("yellow"), textShadow: "4px 4px 0 #FF3EA5" }}>
          DAILY VAULT
        </h1>
        <p style={{ margin: "16px auto 0", maxWidth: 520, color: v("muted") }}>
          One secret, the same for everyone today. {max} messages. Fewest messages tops the board.
        </p>
      </section>

      <StatsTicker />

      <section style={{ ...panel, border: `3px solid ${v("cyan")}` }} aria-label="Daily Vault chat">
        <div className="font-display" style={{ display: "flex", justifyContent: "space-between", gap: 12, fontSize: 11, color: v("cyan") }}>
          <span>BOSS 6 · DAILY VAULT</span>
          <span data-testid="daily-left" style={{ color: v("yellow") }}>{remaining} LEFT</span>
        </div>
        {!nick || !sid ? (
          <form onSubmit={onStart} style={{ display: "flex", flexWrap: "wrap", gap: 10, marginTop: 18 }}>
            <label htmlFor="daily-nick" style={{ position: "absolute", left: -9999 }}>Player name (3-20 characters)</label>
            <input id="daily-nick" value={nickDraft} maxLength={20} placeholder="PLAYER 1" onChange={(e) => setNickDraft(e.target.value)} style={input} />
            <button type="submit" disabled={busy} className="font-display" style={{ ...fire, fontSize: 12 }}>START</button>
          </form>
        ) : (
          <>
            <div data-testid="daily-chat" aria-live="polite" style={{ display: "flex", flexDirection: "column", gap: 12, marginTop: 18 }}>
              {lines.length === 0 && <p style={{ margin: 0, color: v("muted") }}>Playing as {nick}. Make the vault say its code.</p>}
              {lines.map((l, i) =>
                l.who === "guard" ? (
                  <p key={i} style={{ margin: 0, alignSelf: "center", background: v("pink"), color: "#2A0A3F", borderRadius: 999, padding: "8px 16px", fontWeight: 700 }}>{l.text}</p>
                ) : (
                  <p
                    key={i}
                    style={{
                      margin: 0, maxWidth: "85%", padding: "12px 16px", whiteSpace: "pre-wrap", overflowWrap: "anywhere",
                      ...(l.who === "you"
                        ? { alignSelf: "flex-end", background: v("cyan"), color: "#10213A", borderRadius: "18px 18px 4px 18px" }
                        : { alignSelf: "flex-start", background: v("raised"), borderRadius: "18px 18px 18px 4px" }),
                    }}
                  >
                    {l.text}
                  </p>
                ),
              )}
              {waking && <p style={{ margin: 0, color: v("muted") }}>The guard is waking up, hold on...</p>}
            </div>
            {won ? (
              <div style={{ marginTop: 20, display: "flex", flexDirection: "column", gap: 12 }}>
                <p className="font-display" style={{ margin: 0, color: v("green"), fontSize: 14 }}>VAULT CRACKED IN {won.msgs} {won.msgs === 1 ? "MESSAGE" : "MESSAGES"}</p>
                <ShareCard boss={6} msgs={won.msgs} pts={won.pts} name={nick} />
              </div>
            ) : (
              <form onSubmit={onFire} style={{ display: "flex", gap: 10, marginTop: 20 }}>
                <label htmlFor="daily-msg" style={{ position: "absolute", left: -9999 }}>Your message</label>
                <input id="daily-msg" value={msg} maxLength={2000} disabled={remaining <= 0} placeholder={remaining <= 0 ? "Out of messages. Back tomorrow." : "Type your move..."} onChange={(e) => setMsg(e.target.value)} style={input} />
                <button type="submit" disabled={busy || remaining <= 0} style={fire}>FIRE</button>
              </form>
            )}
          </>
        )}
        {err && <p role="alert" style={{ margin: "12px 0 0", color: v("pink") }}>{err}</p>}
      </section>

      <section aria-labelledby="daily-board-h">
        <h2 id="daily-board-h" className="font-display" style={{ margin: "0 0 16px", fontWeight: 400, fontSize: 16, color: v("yellow") }}>TODAY&apos;S BOARD</h2>
        {boardErr && <p style={{ color: v("muted") }}>{boardErr}</p>}
        {daily && daily.board.length === 0 && <p style={{ color: v("muted") }}>Nobody has cracked today&apos;s vault yet. Be first.</p>}
        {daily && daily.board.length > 0 && (
          <table data-testid="daily-board" style={{ width: "100%", borderCollapse: "collapse", border: `3px solid ${v("border")}` }}>
            <thead>
              <tr className="font-display" style={{ background: v("raised"), color: v("cyan"), fontSize: 10, textAlign: "left" }}>
                <th scope="col" style={{ padding: "12px 14px", fontWeight: 400 }}>RANK</th>
                <th scope="col" style={{ padding: "12px 14px", fontWeight: 400 }}>PLAYER</th>
                <th scope="col" style={{ padding: "12px 14px", fontWeight: 400 }}>MSGS</th>
              </tr>
            </thead>
            <tbody>
              {daily.board.map((b) => (
                <tr key={`${b.rank}-${b.at}`} style={{ borderTop: `2px solid ${v("raised")}` }}>
                  <td style={{ padding: "12px 14px", color: b.rank === 1 ? v("yellow") : v("text") }}>{b.rank}</td>
                  <td style={{ padding: "12px 14px", overflowWrap: "anywhere" }}>{b.nickname}</td>
                  <td style={{ padding: "12px 14px" }}>{b.messages}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}
