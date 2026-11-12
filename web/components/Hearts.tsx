import { Heart } from "@/components/Icons";

/** Messages left as 5 SVG hearts (each heart = a fifth of the boss's messages) plus the exact count. */
export default function Hearts({ left, max }: { left: number; max: number }) {
  const full = max > 0 ? Math.ceil((Math.max(0, left) / max) * 5) : 0;
  return (
    <span className="inline-flex items-center gap-2 text-yellow" data-testid="hearts" data-left={left}>
      <span className="inline-flex gap-0.5" aria-hidden>
        {[0, 1, 2, 3, 4].map((i) => <Heart key={i} filled={i < full} size={14} />)}
      </span>
      <span className="font-display text-[11px]" aria-hidden>{left} LEFT</span>
      <span className="sr-only">{left} messages left</span>
    </span>
  );
}
