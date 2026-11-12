export const API = (process.env.NEXT_PUBLIC_API_BASE ?? "").replace(/\/$/, "");

/** v1.1 verdict (docs/CONTRACTS.md 7.2). p_attack is null at L1-L2 (no trained guard there). */
export type Verdict = {
  decision: "BLOCKED" | "PASSED";
  p_attack: number | null;
  category: "injection" | "jailbreak" | "exfiltration" | "keyword" | "none" | string;
  layer: "none" | "regex" | "input_guard" | "output_filter" | string;
};

/** POST /api/message. The v1 fields always exist; the v1.1 fields are optional so the UI also works against a v1 gateway. */
export type Reply = {
  blocked: boolean; block_category: string | null; reply: string; leaked: boolean;
  level_cleared: boolean; messages_left: number; guard_version: string | null; latency_ms: number;
  verdict?: Verdict; points_awarded?: number; total_points?: number; streak?: number;
};
export type LevelInfo = { id: number; name: string; demonstrates: string };
export type Started = { session_id: string; consent: string; levels: LevelInfo[] };

export type LevelStatus = "locked" | "open" | "cleared" | "skipped";
/** GET /api/session/{id} level entry (7.1). */
export type LevelState = { id: number; boss?: string; status: LevelStatus; messages_used: number; points: number; hints_used: number[] };
/** GET /api/session/{id}: v1.1 shape, plus the v1 fields (`cleared`, `messages_used` map, `max_messages`) a v1 gateway returns. */
export type SessionState = {
  levels?: LevelState[]; total_points?: number; streak?: number; hint_cost?: number;
  cleared?: number; messages_used?: Record<string, number>; max_messages?: number; consent?: string;
};
export type HintOut = { level: number; tier: number; text: string; cost: number; total_points: number; hints_used: number[] | number };
export type SkipOut = {
  level: number; status: "skipped"; next_level: number | null;
  how_others_beat_it: { prompt: string; messages: number }[]; fallback_hint: string | null; total_points: number; streak: number;
};
/** GET /api/leaderboard entry: v1.1 (7.6) with the v1 fields as fallbacks. */
export type BoardRow = {
  rank: number; nickname: string; beaten?: number; skipped?: number; points?: number; messages: number;
  levels?: LevelStatus[]; guard_version: string | null; level?: number; seconds?: number;
};
export type Hall = { nickname: string; level: number; guard_version: string | null; at: string };
export type Board = { leaderboard: BoardRow[]; hall_of_breaches: Hall[] };

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

/** Fire-and-forget: wakes the guard host(s) so the instance is warm by the time the player types (ADR-0085). Never throws. */
export function warm(): void {
  try { void fetch(`${API}/api/warm`, { keepalive: true }).catch(() => { /* best effort */ }); } catch { /* best effort */ }
}

let lastPrewarm = 0;
/** Pre-warm on boss hover/focus for bosses 3-5 (at most one call per 8 s); same fire-and-forget request as warm(). */
export function prewarm(): void {
  const now = Date.now();
  if (now - lastPrewarm < 8000) return;
  lastPrewarm = now;
  warm();
}

/** The gateway answers 202 {status:"warming"} while the guard boots; the UI waits and retries up to this many times (~60 s). */
export const MAX_WARMING_TRIES = 12;

export async function call<T>(path: string, init?: RequestInit, onSlow?: () => void, onWarming?: () => void): Promise<T> {
  const slow = onSlow ? setTimeout(onSlow, 1500) : undefined;
  try {
    let warming = 0;
    for (let attempt = 0; ; attempt++) {
      const r = await fetch(`${API}${path}`, { ...init, headers: { "content-type": "application/json", ...(init?.headers ?? {}) } });
      if (r.status === 202) {
        if (warming++ >= MAX_WARMING_TRIES) throw new ApiError(503, "guard waking up");
        onWarming?.();
        let wait = 5;
        try { wait = Math.max(1, Number((await r.json()).retry_after_s) || 5); } catch { /* default */ }
        await new Promise((res) => setTimeout(res, wait * 1000));
        continue;
      }
      if (r.status === 503 && attempt === 0) { await new Promise((res) => setTimeout(res, 2500)); continue; }
      if (!r.ok) {
        let detail = r.statusText;
        try { detail = (await r.json()).detail ?? detail; } catch { /* keep status text */ }
        throw new ApiError(r.status, String(detail));
      }
      return (await r.json()) as T;
    }
  } finally { if (slow) clearTimeout(slow); }
}
