"use client";
import { useEffect, useState } from "react";
import { Speaker } from "@/components/Icons";
import { HI_EVENT, fmt, store } from "@/lib/game";
import { SOUND_EVENT, setSoundOn } from "@/lib/sound";

/** Header HUD: CREDITS ∞, the player's HI score (localStorage) and the sound toggle (default OFF). */
export default function Hud() {
  const [hi, setHi] = useState(0);
  const [sound, setSound] = useState(false);
  useEffect(() => {
    setHi(store.hi());
    setSound(store.sound());
    const onHi = () => setHi(store.hi());
    const onSound = () => setSound(store.sound());
    window.addEventListener(HI_EVENT, onHi);
    window.addEventListener(SOUND_EVENT, onSound);
    window.addEventListener("storage", onHi);
    return () => { window.removeEventListener(HI_EVENT, onHi); window.removeEventListener(SOUND_EVENT, onSound); window.removeEventListener("storage", onHi); };
  }, []);
  return (
    <div className="flex items-center gap-3 font-display text-[10px] text-yellow sm:text-[11px]">
      <span>CREDITS ∞</span>
      <span className="text-cyan" data-testid="hi-score">HI {fmt(hi)}</span>
      <button type="button" onClick={() => setSoundOn(!sound)} aria-pressed={sound} aria-label={sound ? "Sound on" : "Sound off"}
        data-testid="sound-toggle"
        className="inline-flex h-11 w-11 items-center justify-center rounded-xl border-[3px] border-border-ui text-text hover:border-muted">
        <Speaker on={sound} size={18} />
      </button>
    </div>
  );
}
