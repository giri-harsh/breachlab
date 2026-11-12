"use client";
import { useGuardStatus } from "@/lib/guard";

/** Shown only while the guard is DOWN (gateway guard_state "down"): bosses 3-5 fail closed (ADR-0050), bosses 1-2 are unaffected. A cold guard is "waking up", never this banner. */
export default function GuardBanner() {
  const { retry } = useGuardStatus();
  return (
    <div data-testid="levels-offline-banner" role="note" className="mb-4 rounded-[18px] border-[3px] border-yellow bg-surface p-4 text-sm text-soft">
      <p className="font-display text-[11px] leading-5 text-yellow">Levels 3-5 are temporarily offline.</p>
      <p className="mt-2">
        Bosses 3 to 5 need the guard model, which is not answering right now. It fails closed: nothing is let through unchecked. Levels 1 and 2 are
        fully playable, and the high scores, How it works and Model card pages work as usual.
      </p>
      <button type="button" onClick={retry} data-testid="guard-retry-banner" className="btn mt-3 min-h-11 font-display text-[10px]">RETRY</button>
    </div>
  );
}
