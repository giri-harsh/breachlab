"use client";
// Tiny WebAudio blips. Off by default; the HUD toggle stores the choice (bl.sound). No audio files, no library.
import { store } from "@/lib/game";

let ctx: AudioContext | null = null;
export const SOUND_EVENT = "bl:sound";

export function soundOn(): boolean { return store.sound(); }
export function setSoundOn(on: boolean): void {
  store.setSound(on);
  try { window.dispatchEvent(new CustomEvent(SOUND_EVENT, { detail: on })); } catch { /* ignore */ }
  if (on) blip("ok");
}

const TUNES: Record<"fire" | "ok" | "block" | "win" | "lose", [number, number][]> = {
  fire: [[660, 0.05]],
  ok: [[520, 0.06], [780, 0.06]],
  block: [[220, 0.09], [160, 0.12]],
  win: [[523, 0.08], [659, 0.08], [784, 0.08], [1047, 0.16]],
  lose: [[392, 0.12], [330, 0.12], [262, 0.2]],
};

export function blip(kind: keyof typeof TUNES): void {
  if (!soundOn()) return;
  try {
    const AC = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AC) return;
    ctx = ctx ?? new AC();
    let t = ctx.currentTime;
    for (const [freq, dur] of TUNES[kind]) {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "square";
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.06, t);
      gain.gain.exponentialRampToValueAtTime(0.0001, t + dur);
      osc.connect(gain).connect(ctx.destination);
      osc.start(t);
      osc.stop(t + dur);
      t += dur;
    }
  } catch { /* audio unavailable */ }
}
