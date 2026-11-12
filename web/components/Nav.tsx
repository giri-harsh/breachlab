import Link from "next/link";
import GuardStatus from "@/components/GuardStatus";
import Hud from "@/components/Hud";

const links = [["/play", "Play"], ["/leaderboard", "High scores"], ["/under-the-hood", "How it works"], ["/model-card", "Model card"]];
export default function Nav() {
  return (
    <header className="mx-auto max-w-[1240px] px-4 pt-4 sm:px-8 sm:pt-6">
      <nav aria-label="Main" className="flex flex-wrap items-center gap-x-6 gap-y-2">
        <Link href="/" className="font-display text-[15px] text-pink no-underline hover:text-pink sm:text-lg" aria-label="BreachLab home">
          BREACH<span className="text-cyan">LAB</span>
        </Link>
        <span className="ml-auto flex items-center gap-2 sm:order-last sm:ml-0">
          <Hud />
        </span>
        <ul className="flex w-full flex-wrap items-center gap-x-5 font-medium sm:ml-auto sm:w-auto sm:gap-x-7">
          {links.map(([href, label]) => (
            <li key={href}><Link href={href} className="inline-flex min-h-11 items-center text-text no-underline hover:text-yellow">{label}</Link></li>
          ))}
          <li><GuardStatus /></li>
        </ul>
      </nav>
    </header>
  );
}
