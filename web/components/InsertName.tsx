/** The /play cabinet shown before a run exists. Also the loading shell, so the page does not shift when the form appears. */
export default function InsertName({ children }: { children?: React.ReactNode }) {
  return (
    <section className="cabinet grid-bg mx-auto max-w-3xl px-5 py-12 text-center sm:px-10" aria-labelledby="ins">
      <h1 id="ins" className="font-display text-lg leading-8 text-yellow sm:text-2xl">INSERT NAME</h1>
      <p className="mx-auto mt-4 max-w-md text-soft">Pick a player name to start a run. No signup; your progress stays in this browser.</p>
      <div className="mt-8">{children ?? <div aria-hidden className="h-[136px] sm:h-[60px]" />}</div>
    </section>
  );
}
