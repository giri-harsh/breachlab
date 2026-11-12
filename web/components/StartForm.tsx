"use client";
import { useEffect, useState } from "react";
import { ErrorLine, Offline } from "@/components/Alerts";
import { PlayIcon } from "@/components/Icons";
import { ApiError, call, warm, type Started } from "@/lib/api";
import { store, type SavedSession } from "@/lib/game";

/** Player name + START. Creates the session (POST /api/session), saves it, wakes the guard (ADR-0085). */
export default function StartForm({ onStarted, autoFocus = false, label = "START" }: {
  onStarted: (s: SavedSession) => void; autoFocus?: boolean; label?: string;
}) {
  const [nick, setNick] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [offline, setOffline] = useState<"" | "guard" | "vault">("");
  useEffect(() => { setNick((n) => n || store.name()); }, []);

  async function start(e: React.FormEvent) {
    e.preventDefault(); setError(""); setOffline(""); setBusy(true);
    try {
      const s = await call<Started>("/api/session", { method: "POST", body: JSON.stringify({ nickname: nick.trim() }) });
      const saved = { id: s.session_id, nickname: nick.trim(), consent: s.consent };
      store.setSession(saved); store.setName(nick.trim());
      warm();
      onStarted(saved);
    } catch (err) {
      if (err instanceof ApiError && err.status === 503) setOffline(/vault/i.test(err.message) ? "vault" : "guard");
      else setError(err instanceof ApiError ? err.message : "Could not reach the game server.");
    } finally { setBusy(false); }
  }

  return (
    <form onSubmit={start} className="flex flex-col items-center gap-4" data-testid="start-form">
      <div className="flex w-full flex-wrap justify-center gap-4">
        <label htmlFor="nick" className="sr-only">Player name (3-20 characters)</label>
        <input id="nick" value={nick} onChange={(e) => setNick(e.target.value)} maxLength={20} minLength={3} required autoFocus={autoFocus}
          placeholder="PLAYER 1" autoComplete="nickname"
          className="min-h-[60px] w-full max-w-[280px] rounded-[14px] border-[3px] border-cyan bg-surface px-4 text-center font-display text-[13px] text-text placeholder:text-muted-2 focus:outline-none focus:ring-4 focus:ring-yellow" />
        <button disabled={busy} className="btn btn-yellow min-h-[60px] px-8 font-display text-sm">
          <PlayIcon size={14} />{busy ? "LOADING…" : label}
        </button>
      </div>
      {error && <ErrorLine text={error} />}
      {offline && <Offline kind={offline} />}
    </form>
  );
}
