"use client";
// Share button for a defeated boss (CONTRACTS.md 7.9). Lane V. Lane U imports it as-is.
// navigator.share when the device has it, else copy the /share link; plus an X (Twitter) intent link.
import { useState } from "react";
import { cardQuery, headline, parseCard, v } from "@/app/api/og/card";

export type ShareCardProps = { boss: number; msgs: number; pts: number; name: string };

async function copy(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand("copy");
      document.body.removeChild(ta);
      return ok;
    } catch {
      return false;
    }
  }
}

const btn: React.CSSProperties = {
  minHeight: 48, minWidth: 44, padding: "0 20px", borderRadius: 14, display: "inline-flex", alignItems: "center",
  justifyContent: "center", gap: 10, fontSize: 13, cursor: "pointer", textDecoration: "none",
};

export default function ShareCard(props: ShareCardProps) {
  const card = parseCard({ boss: String(props.boss), msgs: String(props.msgs), pts: String(props.pts), name: props.name });
  const [status, setStatus] = useState("");
  const path = `/share?${cardQuery(card)}`;
  const text = `${headline(card)} on BreachLab. Can you beat my score?`;
  const url = () => `${window.location.origin}${path}`;

  async function onShare() {
    const link = url();
    try {
      if (typeof navigator.share === "function") {
        await navigator.share({ title: "BreachLab", text, url: link });
        setStatus("SHARED");
        return;
      }
    } catch (e) {
      if ((e as { name?: string })?.name === "AbortError") return; // the player closed the share sheet
    }
    setStatus((await copy(link)) ? "LINK COPIED" : `COPY THIS LINK: ${link}`);
  }

  const siteOrigin = "https://breachlab-web.vercel.app";
  const intent = "https://twitter.com/intent/tweet" + "?" + new URLSearchParams({ text, url: siteOrigin + path }).toString();
  return (
    <div data-testid="share-card" style={{ display: "flex", flexWrap: "wrap", gap: 12, alignItems: "center" }}>
      <button
        type="button"
        onClick={() => void onShare()}
        aria-label={`Share: ${headline(card).toLowerCase()}`}
        className="font-display"
        style={{ ...btn, border: 0, background: v("yellow"), color: "#2A0A3F", boxShadow: "0 4px 0 #B89A1A" }}
      >
        <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
          <path fill="currentColor" d="M18 16a3 3 0 0 0-2.4 1.2l-6.7-3.4a3 3 0 0 0 0-1.6l6.7-3.4A3 3 0 1 0 15 7l-6.7 3.4a3 3 0 1 0 0 3.2L15 17a3 3 0 1 0 3-1z" />
        </svg>
        SHARE
      </button>
      <a
        href={intent}
        target="_blank"
        rel="noopener noreferrer"
        aria-label="Post your result on X (opens in a new tab)"
        className="font-display"
        style={{ ...btn, border: `3px solid ${v("cyan")}`, color: v("cyan"), background: "transparent" }}
      >
        <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
          <path fill="currentColor" d="M4 3h4.6l4.1 5.8L17.8 3H20l-6.3 7.3L21 21h-4.6l-4.4-6.3L6.5 21H4.3l6.7-7.7z" />
        </svg>
        POST
      </a>
      <p role="status" aria-live="polite" style={{ margin: 0, minHeight: 20, fontSize: 14, color: v("green"), wordBreak: "break-all" }}>
        {status}
      </p>
    </div>
  );
}
