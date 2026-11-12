"use client";
import { useCallback, useSyncExternalStore } from "react";
import { API, warm } from "@/lib/api";

/**
 * Guard status is RUNTIME only (no build-time flag). The gateway's /api/health reports guard_state:
 *  ok -> "online", warming (guard is cold, boots in ~10 s) -> "warming", down -> "offline".
 * "warming" never locks a boss: bosses 3-5 show BOSS WAKING UP… and unlock by themselves.
 */
export type GuardState = "checking" | "online" | "warming" | "offline";

/** Levels 3-5 need the trained guard (ADR-0050: they fail closed with 503 when it is down). */
export const GUARD_LEVELS = [3, 4, 5];

export const POLL_MS = 2000;
export const WARMING_MAX_MS = 30_000;

type Health = { ok?: boolean; guard?: boolean; guard_ok?: boolean; guard_state?: string };

/** Maps a /api/health body to a state. An older gateway without guard_state: a false `guard_ok` is treated as "warming", never "offline". */
export function stateFromHealth(d: Health): GuardState {
  if (!d.ok || d.guard === false) return "offline";
  if (d.guard_state === "ok") return "online";
  if (d.guard_state === "warming") return "warming";
  if (d.guard_state === "down") return "offline";
  return d.guard_ok === false ? "warming" : "online";
}

export type GuardStatus = { state: GuardState; expired: boolean };

let status: GuardStatus = { state: "checking", expired: false };
let warmingSince = 0;
let answered = false; // the gateway has reported a guard state at least once in this window
let timer: ReturnType<typeof setTimeout> | null = null;
let started = false;
let epoch = 0;
const listeners = new Set<() => void>();

function set(next: GuardStatus) {
  if (next.state === status.state && next.expired === status.expired) return;
  status = next;
  listeners.forEach((l) => l());
}

/** The gateway is a serverless function too: its own cold start can take 10 s or more, so one attempt waits up to 20 s. */
export const HEALTH_TIMEOUT_MS = 20_000;

/**
 * One GET /api/health. A network error, timeout or 502/504 means the gateway has not answered YET (it may itself be
 * cold): that is reported as "warming" and only becomes "offline" when the 30 s polling window ends without an answer.
 */
export async function fetchGuardState(): Promise<{ state: GuardState; reached: boolean }> {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), HEALTH_TIMEOUT_MS);
  try {
    const r = await fetch(`${API}/api/health`, { signal: ctl.signal, cache: "no-store" });
    if (r.ok) return { state: stateFromHealth((await r.json()) as Health), reached: true };
    // 502/504 come from the platform while the gateway function boots: not answered yet. Any other error status
    // (e.g. the gateway's own 503) is an answer: the guard path is down.
    return r.status === 502 || r.status === 504 ? { state: "warming", reached: false } : { state: "offline", reached: true };
  } catch {
    return { state: "warming", reached: false };
  } finally {
    clearTimeout(t);
  }
}

async function tick(my: number) {
  const { state: s, reached } = await fetchGuardState();
  if (my !== epoch) return;
  if (reached) answered = true;
  if (s === "warming") {
    if (!warmingSince) warmingSince = Date.now();
    const expired = Date.now() - warmingSince >= WARMING_MAX_MS;
    // Poll every 2 s for up to 30 s, then stop. A guard that answered "warming" is let through (the message path has
    // its own 202 retry); a gateway that never answered at all is offline.
    set({ state: expired && !answered ? "offline" : "warming", expired });
    if (!expired) timer = setTimeout(() => void tick(my), POLL_MS);
    return;
  }
  warmingSince = 0;
  set({ state: s, expired: false });
}

function start() {
  if (started) return;
  started = true;
  void tick(epoch);
}

/** Retry button / manual re-check: wake the guard again and restart the 30 s polling window. */
export function retryGuard() {
  epoch += 1;
  if (timer) clearTimeout(timer);
  warmingSince = 0; answered = false;
  set({ state: "checking", expired: false });
  warm();
  started = true;
  void tick(epoch);
}

/** Test hook: forget everything (a fresh page load does this implicitly). */
export function resetGuardStore() {
  epoch += 1;
  if (timer) clearTimeout(timer);
  started = false; warmingSince = 0; answered = false;
  status = { state: "checking", expired: false };
}

function subscribe(cb: () => void) {
  listeners.add(cb);
  start();
  return () => { listeners.delete(cb); };
}
const snapshot = () => status;
const serverSnapshot = (): GuardStatus => ({ state: "checking", expired: false });

/** Shared runtime guard status: polls /api/health every 2 s while "warming" (max 30 s), once otherwise. */
export function useGuardStatus(): GuardStatus & { retry: () => void } {
  const s = useSyncExternalStore(subscribe, snapshot, serverSnapshot);
  const retry = useCallback(() => retryGuard(), []);
  return { ...s, retry };
}

export function useGuardState(): GuardState {
  return useGuardStatus().state;
}
