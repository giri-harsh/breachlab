import type { Metadata, Viewport } from "next";
import { Press_Start_2P, Space_Grotesk } from "next/font/google";
import "./globals.css";
import Nav from "@/components/Nav";

const display = Press_Start_2P({ weight: "400", subsets: ["latin"], variable: "--font-display", display: "swap" });
const body = Space_Grotesk({ weight: ["400", "500", "700"], subsets: ["latin"], variable: "--font-body", display: "swap" });

export const metadata: Metadata = {
  title: "BreachLab · Break the bot",
  description: "Five bosses guard a secret code. Trick the AI into leaking it. A research demo of a fine-tuned prompt-injection guard.",
};
export const viewport: Viewport = { width: "device-width", initialScale: 1, themeColor: "#12062B" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable}`}>
      <body className="font-body min-h-screen antialiased">
        <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-yellow focus:px-4 focus:py-2 focus:text-ink">Skip to content</a>
        <Nav />
        <main id="main" className="mx-auto max-w-[1240px] px-4 py-6 sm:px-8">{children}</main>
        <footer className="mx-auto max-w-[1240px] px-4 pb-10 text-[13px] text-muted-2 sm:px-8">
          Research demo. Not a claim of production safety. Prompts are logged and used to train the guard.
        </footer>
      </body>
    </html>
  );
}
