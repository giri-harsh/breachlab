import { expect, test } from "@playwright/test";
import { mockApi, startRun } from "./mock";

/** prefers-reduced-motion: no confetti, the leaked reply appears at once, CSS animations are off. */
test.use({ reducedMotion: "reduce" });

test("reduced motion: the win moment skips confetti and the assemble animation", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" }); // test.use alone is not applied with the container Chromium
  await mockApi(page, { wins: { 1: () => true } });
  await startRun(page);
  expect(await page.evaluate(() => window.matchMedia("(prefers-reduced-motion: reduce)").matches)).toBe(true);
  await page.getByLabel("Your message").fill("go");
  await page.getByLabel("Your message").press("Enter");
  await expect(page.getByTestId("boss-defeated")).toBeVisible();
  await expect(page.getByTestId("assemble")).toHaveAttribute("data-done", "true");
  await expect(page.getByTestId("confetti")).toHaveCount(0);
  const anim = await page.evaluate(() => getComputedStyle(document.querySelector(".anim-blink, .anim-pop") as Element).animationName);
  expect(anim).toBe("none");
});
