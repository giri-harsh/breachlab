"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { API } from "@/lib/api";

type Item = { id: number; message: string; verdict: string | null; category: string | null; confidence: number | null; leaked: boolean; level: number };
const KEYS: Record<string, [string, string | null]> = { s: ["SAFE", null], i: ["UNSAFE", "injection"], j: ["UNSAFE", "jailbreak"], e: ["UNSAFE", "exfiltration"] };

export default function Review() {
  const [pw, setPw] = useState("");
  const [authed, setAuthed] = useState(false);
  const [items, setItems] = useState<Item[]>([]);
  const [idx, setIdx] = useState(0);
  const [msg, setMsg] = useState("");
  const history = useRef<number[]>([]);

  const headers = useCallback(() => ({ "content-type": "application/json", authorization: `Bearer ${pw}` }), [pw]);
  async function load(e?: React.FormEvent) {
    e?.preventDefault();
    const r = await fetch(`${API}/api/admin/queue?limit=50`, { headers: headers() });
    if (!r.ok) { setMsg("Wrong password or server error."); return; }
    setItems(await r.json()); setIdx(0); setAuthed(true); setMsg("");
  }
  const label = useCallback(async (key: string) => {
    const it = items[idx]; if (!it) return;
    const [lab, cat] = KEYS[key];
    await fetch(`${API}/api/admin/label`, { method: "POST", headers: headers(), body: JSON.stringify({ attempt_id: it.id, label: lab, category: cat }) });
    history.current.push(idx); setIdx((i) => i + 1);
  }, [items, idx, headers]);

  useEffect(() => {
    if (!authed) return;
    const h = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === "INPUT") return;
      if (e.key in KEYS) void label(e.key);
      else if (e.key === "k") setIdx((i) => i + 1);
      else if (e.key === "u") { const p = history.current.pop(); if (p !== undefined) setIdx(p); }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [authed, label]);

  if (!authed) {
    return (
      <form onSubmit={load} className="cabinet mx-auto flex max-w-md flex-col gap-4 bg-surface p-6">
        <h1 className="font-display text-base leading-7 text-cyan">REVIEW QUEUE</h1>
        <label htmlFor="pw" className="text-sm text-muted">Admin password</label>
        <input id="pw" type="password" value={pw} onChange={(e) => setPw(e.target.value)} className="field" />
        <button className="btn btn-yellow font-display text-xs">OPEN</button>
        {msg && <p role="alert" className="text-yellow">{msg}</p>}
      </form>
    );
  }
  const it = items[idx];
  return (
    <section>
      <h1 className="font-display text-base leading-7 text-cyan">REVIEW QUEUE ({Math.min(idx, items.length)}/{items.length})</h1>
      <p className="mt-2 text-sm text-muted">Keys: s safe, i injection, j jailbreak, e exfiltration, k skip, u undo.</p>
      {it ? (
        <div className="mt-4 rounded-[20px] border-[3px] border-border bg-surface p-5">
          <p className="text-xs text-muted">L{it.level} - guard said {it.verdict ?? "n/a"} {it.category ?? ""} ({it.confidence?.toFixed(2) ?? "-"}) {it.leaked ? "- LEAKED" : ""}</p>
          <p className="mt-2 whitespace-pre-wrap break-words text-lg">{it.message}</p>
        </div>
      ) : <p className="mt-4 text-muted">Queue empty.</p>}
    </section>
  );
}
