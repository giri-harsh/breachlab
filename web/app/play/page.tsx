import { Suspense } from "react";
import InsertName from "@/components/InsertName";
import PlayGame from "./PlayGame";

export const metadata = { title: "Play · BreachLab" };

export default function PlayPage() {
  return (
    <Suspense fallback={<InsertName />}>
      <PlayGame />
    </Suspense>
  );
}
