// Inline SVG icons (no emoji, no icon font). Pixel-ish shapes on a 16x16 grid; decorative unless a title is given.
type P = { className?: string; title?: string; size?: number };

function Svg({ className, title, size = 16, children }: P & { children: React.ReactNode }) {
  return (
    <svg viewBox="0 0 16 16" width={size} height={size} className={className} fill="currentColor" shapeRendering="crispEdges"
      role={title ? "img" : undefined} aria-hidden={title ? undefined : true} aria-label={title}>
      {children}
    </svg>
  );
}

export function Heart({ filled = true, ...p }: P & { filled?: boolean }) {
  return (
    <Svg {...p}>
      {filled
        ? <path d="M2 3h4v1h1v1h2V4h1V3h4v1h1v5h-1v1h-1v1h-1v1h-1v1h-1v1H7v-1H6v-1H5v-1H4v-1H3V9H2V8H1V4h1z" />
        : <path fillRule="evenodd" d="M2 3h4v1h1v1h2V4h1V3h4v1h1v5h-1v1h-1v1h-1v1h-1v1h-1v1H7v-1H6v-1H5v-1H4v-1H3V9H2V8H1V4h1zm1 2v3h1v1h1v1h1v1h1v1h2v-1h1v-1h1V9h1V8h1V5h-3v1H9v1H7V6H6V5z" />}
    </Svg>
  );
}
export const PlayIcon = (p: P) => <Svg {...p}><path d="M4 2h2v1h2v1h2v1h2v1h1v4h-1v1h-2v1H8v1H6v1H4z" /></Svg>;
export const Star = (p: P) => <Svg {...p}><path d="M7 1h2v4h5v2h-1v1h-1v1h-1v2h1v3h-2v-1H9v-1H7v1H6v1H4v-3h1V9H4V8H3V7H2V5h5z" /></Svg>;
export const Lock = (p: P) => <Svg {...p}><path fillRule="evenodd" d="M5 2h6v1h1v4h1v8H3V7h1V3h1zm1 2v3h4V4zm1 6v3h2v-3z" /></Svg>;
export const Bolt = (p: P) => <Svg {...p}><path d="M9 1h3L9 7h4l-7 8 2-6H4z" /></Svg>;
export const Skip = (p: P) => <Svg {...p}><path d="M2 3h2v1h1v1h1v1h1v4H6v1H5v1H4v1H2zM8 3h2v1h1v1h1v1h1v4h-1v1h-1v1h-1v1H8zM14 3h1v10h-1z" /></Svg>;
export const Check = (p: P) => <Svg {...p}><path d="M13 3h2v2h-1v1h-1v1h-1v1h-1v1h-1v1H9v1H8v1H7v1H5v-1H4v-1H3v-1H2V9H1V7h2v1h1v1h1v1h1V9h1V8h1V7h1V6h1V5h1V4h1z" /></Svg>;
export const Coin = (p: P) => <Svg {...p}><path fillRule="evenodd" d="M5 1h6v1h2v2h1v8h-1v2h-2v1H5v-1H3v-2H2V4h1V2h2zm2 3v8h2V4z" /></Svg>;
export const Trophy = (p: P) => <Svg {...p}><path d="M3 1h10v2h2v4h-1v1h-2v1h-1v1H9v2h2v2H5v-2h2v-2H6V9H5V8H3V7H2V6H1V3h2zm0 3v2h1V4zm9 0v2h1V4z" /></Svg>;
export const Speaker = ({ on, ...p }: P & { on: boolean }) => (
  <Svg {...p}>
    <path d="M1 6h3l4-4h1v12H8l-4-4H1z" />
    {on ? <path d="M11 5h1v1h1v4h-1v1h-1zM13 3h1v1h1v8h-1v1h-1z" /> : <path d="M10 5h1v1h1v1h1V6h1V5h1v1h-1v1h-1v2h1v1h1v1h-1v-1h-1V9h-1v1h-1v1h-1v-1h1V9h1V7h-1V6h-1z" />}
  </Svg>
);
export const Skull = (p: P) => <Svg {...p}><path fillRule="evenodd" d="M4 1h8v1h1v1h1v6h-1v2h-2v3H5v-3H3V9H2V3h1V2h1zm1 5v2h2V6zm4 0v2h2V6zm-2 3v1h2V9z" /></Svg>;
export const Arrow = (p: P) => <Svg {...p}><path d="M1 7h9V4h1v1h1v1h1v1h1v2h-1v1h-1v1h-1v1h-1V9H1z" /></Svg>;
