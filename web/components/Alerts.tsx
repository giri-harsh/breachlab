import { Skull } from "@/components/Icons";

/** Fail-closed messages (ADR-0050). The testids are used by the offline, play and live specs. */
export function Offline({ kind }: { kind: "guard" | "vault" }) {
  if (kind === "vault") {
    return (
      <p role="alert" data-testid="vault-busy" className="rounded-[16px] border-[3px] border-yellow bg-surface p-3 text-sm text-soft">
        <span className="font-bold text-yellow">The vault assistant is busy</span> (its free-tier model is rate limited). The attempt was not counted; try again in a few seconds.
      </p>
    );
  }
  return (
    <p role="alert" data-testid="guard-offline" className="flex gap-3 rounded-[16px] border-[3px] border-pink bg-surface p-3 text-sm text-soft">
      <Skull className="mt-0.5 shrink-0 text-pink" size={18} />
      <span><span className="font-bold text-pink">Guard offline.</span> The guard service is waking up or unavailable (it fails closed, so nothing is let through). Levels 1 and 2 do not need it; try again in a minute.</span>
    </p>
  );
}

export function ErrorLine({ text }: { text: string }) {
  return <p role="alert" className="rounded-[14px] border-[3px] border-pink bg-surface px-3 py-2 text-sm text-soft">{text}</p>;
}
