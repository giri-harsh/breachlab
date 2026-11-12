import type { BoardRow, LevelStatus } from "@/lib/api";
import { fmt, ordinal } from "@/lib/game";

/** v1.1 rows carry `levels` (status x5); a v1 row only has `level` (highest cleared), so 1..level count as cleared. */
function statuses(r: BoardRow): LevelStatus[] {
  if (Array.isArray(r.levels) && r.levels.length) return r.levels;
  const n = typeof r.level === "number" ? r.level : 0;
  return [1, 2, 3, 4, 5].map((i) => (i <= n ? "cleared" : "locked"));
}

function Chip({ n, st }: { n: number; st: LevelStatus }) {
  if (st === "cleared") return <span className="chip bg-green text-ink" title={`Boss ${n} beaten`}>{n}</span>;
  if (st === "skipped") return <span className="chip bg-chip-grey-bg text-chip-grey" title={`Boss ${n} skipped`} data-testid="skip-chip">SKIP</span>;
  return <span className="chip border-2 border-border text-muted-2" title={`Boss ${n} not beaten`}>{n}</span>;
}

/** HIGH SCORES table, in the order the API returns (beaten desc, points desc, messages asc). */
export default function HighScores({ rows, limit }: { rows: BoardRow[]; limit?: number }) {
  const shown = typeof limit === "number" ? rows.slice(0, limit) : rows;
  return (
    <div className="overflow-hidden rounded-[24px] border-[3px] border-border">
      <table className="w-full text-left" data-testid="high-scores">
        <caption className="sr-only">High scores: bosses beaten, then points, then fewest messages</caption>
        <thead className="bg-raised font-display text-[10px] text-cyan">
          <tr>
            <th scope="col" className="px-4 py-4 font-normal sm:px-6">RANK</th>
            <th scope="col" className="px-2 py-4 font-normal">PLAYER</th>
            <th scope="col" className="hidden px-2 py-4 font-normal md:table-cell">BOSSES</th>
            <th scope="col" className="px-2 py-4 text-right font-normal">SCORE</th>
            <th scope="col" className="hidden px-4 py-4 text-right font-normal sm:table-cell sm:px-6">MSGS</th>
          </tr>
        </thead>
        <tbody>
          {shown.map((r) => {
            const st = statuses(r);
            const beaten = r.beaten ?? st.filter((s) => s === "cleared").length;
            return (
              <tr key={`${r.rank}-${r.nickname}`} className="border-t-2 border-raised align-top" data-testid="score-row">
                <td className={`px-4 py-4 font-display text-[11px] leading-6 sm:px-6 ${r.rank === 1 ? "text-yellow" : "text-text"}`}>{ordinal(r.rank)}</td>
                <td className="px-2 py-3">
                  <span className="block break-all font-bold">{r.nickname}</span>
                  <span className="mt-1 block text-[13px] text-muted">{beaten}/5 beaten{r.skipped ? ` · ${r.skipped} skipped` : ""}</span>
                  <span className="mt-2 flex flex-wrap gap-1 md:hidden">{st.map((s, i) => <Chip key={i} n={i + 1} st={s} />)}</span>
                </td>
                <td className="hidden px-2 py-4 md:table-cell"><span className="flex gap-1">{st.map((s, i) => <Chip key={i} n={i + 1} st={s} />)}</span></td>
                <td className="px-2 py-4 text-right font-bold text-yellow">{typeof r.points === "number" ? fmt(r.points) : "-"}</td>
                <td className="hidden px-4 py-4 text-right text-muted sm:table-cell sm:px-6">{r.messages}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {shown.length === 0 && <p className="border-t-2 border-raised px-6 py-5 text-muted">No high scores yet. Be PLAYER 1.</p>}
    </div>
  );
}
