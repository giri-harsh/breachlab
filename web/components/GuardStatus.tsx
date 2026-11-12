"use client";
import { useGuardState } from "@/lib/guard";

/** Nav pill from the shared runtime guard status: online, waking up (cold, pulsing, never an error) or offline (guard_state "down"). */
export default function GuardStatus() {
  const state = useGuardState();
  const label = state === "online" ? "guard online" : state === "offline" ? "guard offline" : state === "warming" ? "guard waking up" : "checking guard";
  const tone = state === "online" ? "border-green text-green" : state === "offline" ? "border-pink text-pink" : state === "warming" ? "border-yellow text-yellow" : "border-border-ui text-muted";
  return (
    <span data-testid="guard-status" data-state={state} role="status"
      className={`inline-flex items-center gap-1.5 rounded-full border-2 bg-surface px-2.5 py-0.5 text-xs font-bold ${tone}`}>
      <span aria-hidden className={`inline-block h-2 w-2 rounded-full bg-current ${state === "warming" ? "animate-pulse" : ""}`} />{label}
    </span>
  );
}
