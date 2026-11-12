import { expect, test, type Page } from "@playwright/test";
import fs from "node:fs";
import { mockApi, noScroll } from "./mock";

/**
 * Guard-offline UX (ADR-0063: no guard host, bosses 3-5 fail closed). Mocked API; screenshots go to
 * web/e2e-artifacts/ (git-ignored).
 */
const ART = "e2e-artifacts";
async function shot(page: Page, name: string, project: string) {
  fs.mkdirSync(ART, { recursive: true });
  await page.screenshot({ path: `${ART}/${project}-${name}.png`, fullPage: true });
}

test.beforeEach(async ({ page }) => {
  await mockApi(page, { wins: { 1: () => true, 2: () => true } });
  // health says "GUARD_URL is set" but the real probe failed: exactly what a placeholder GUARD_URL looks like once the gateway probes it.
  await page.route("**/api/health", (r) => r.fulfill({ json: { ok: true, guard: true, guard_ok: false, guard_state: "down" } }));
});

test("offline: start screen explains levels 3-5 are offline and boss 1 can still be started", async ({ page }, info) => {
  await page.goto("/");
  await expect(page.getByTestId("guard-status")).toHaveAttribute("data-state", "offline");
  const banner = page.getByTestId("levels-offline-banner");
  await expect(banner).toContainText("Levels 3-5 are temporarily offline");
  await expect(banner).toContainText("Levels 1 and 2 are");
  await expect(page.getByRole("button", { name: /start/i })).toBeEnabled();
  expect(await noScroll(page)).toBe(true);
  await shot(page, "start-offline", info.project.name);
});

test("offline: bosses 1 and 2 stay playable; boss 3 shows an offline panel and a disabled composer", async ({ page }, info) => {
  let sent = 0;
  page.on("request", (r) => { if (r.url().endsWith("/api/message")) sent += 1; });
  await page.goto("/");
  await page.getByLabel("Player name (3-20 characters)").fill("Offline Player");
  await page.getByRole("button", { name: /start/i }).click();
  await expect(page.getByTestId("levels-offline-banner")).toBeVisible();
  const tab3 = page.getByTestId("boss-tab-3");
  await expect(tab3).toBeDisabled(); // locked until boss 2 is beaten or skipped, and labelled offline
  await expect(tab3).toHaveAttribute("aria-label", /offline/);
  for (const level of [1, 2]) {
    await expect(page.getByTestId("boss-title")).toContainText(`${level} ·`);
    await page.getByLabel("Your message").fill(`try level ${level}`);
    await page.getByRole("button", { name: "FIRE" }).click();
    await page.getByTestId("boss-defeated").getByRole("button", { name: /NEXT BOSS/ }).click();
  }
  expect(sent).toBe(2);
  await expect(page.getByTestId("level-offline")).toContainText("Level 3 is offline right now");
  await expect(page.getByLabel("Your message")).toBeDisabled();
  await expect(page.getByRole("button", { name: "FIRE" })).toBeDisabled();
  expect(await noScroll(page)).toBe(true);
  await shot(page, "l3-offline", info.project.name);
  await page.getByTestId("boss-tab-2").click();
  await expect(page.getByTestId("level-offline")).toHaveCount(0);
});

for (const [path, heading] of [["/leaderboard", /HIGH SCORES/], ["/under-the-hood", /Scorecard/], ["/model-card", /MODEL CARD/]] as const) {
  test(`offline: ${path} renders`, async ({ page }, info) => {
    await page.goto(path);
    await expect(page.getByRole("heading", { name: heading }).first()).toBeVisible();
    await expect(page.getByTestId("guard-status")).toHaveAttribute("data-state", "offline");
    expect(await noScroll(page)).toBe(true);
    await shot(page, path.replace("/", ""), info.project.name);
  });
}

test("health reporting guard_ok true keeps the banner away", async ({ page }) => {
  await page.route("**/api/health", (r) => r.fulfill({ json: { ok: true, guard: true, guard_ok: true, guard_state: "ok" } }));
  await page.goto("/");
  await expect(page.getByTestId("guard-status")).toHaveAttribute("data-state", "online");
  await expect(page.getByTestId("levels-offline-banner")).toHaveCount(0);
});
