// /daily : the Daily Vault (boss 6, CONTRACTS.md 7.7). Lane V.
import type { Metadata } from "next";
import DailyGame from "./DailyGame";

export const metadata: Metadata = {
  title: "Daily Vault | BreachLab",
  description: "One vault, one secret, the same for everyone today. Ten messages. Fewest messages wins the daily board.",
};

export default function DailyPage() {
  return <DailyGame />;
}
