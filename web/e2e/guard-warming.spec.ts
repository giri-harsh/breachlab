import { expect, test, type Page } from "@playwright/test";
import { mockApi } from "./mock";

/**
 * A COLD guard is "warming", never "offline" (guard_state ok | warming | down from /api/health, runtime only).
 * (a) cold guard (health says warming for 8 s): BOSS WAKING UP… with a pulse, boss 3 is not locked, it opens by
 *     itself and the first message works; (b) guard down: offline panel with a RETRY that recovers.
 */
async function reachBoss3(page: Page) {
  await page.goto("/");
  await page.getByLabel("Player name (3-20 characters)").fill("Cold Guard Player");
  await page.getByRole("button", { name: /start/i }).click();
  for (const level of [1, 2]) {
    await expect(page.getByTestId("boss-title")).toContainText(`${level} ·`);
    await page.getByLabel("Your message").fill(`try level ${level}`);
    await page.getByRole("button", { name: "FIRE" }).click();
    await page.getByTestId("boss-defeated").getByRole("button", { name: /NEXT BOSS/ }).click();
  }
  await expect(page.getByTestId("boss-title")).toContainText("3 ·");
}

test("cold guard: waking screen, no lock, unlocks by itself, first message works", async ({ page }) => {
  const st = await mockApi(page, { wins: { 1: () => true, 2: () => true } });
  const t0 = Date.now();
  let seenWaking = false;
  let healthCalls = 0;
  await page.route(/\/api\/health$/, (r) => {
    healthCalls += 1;
    // cold for 8 s, and at least until the test has seen the waking screen
    const cold = !seenWaking || Date.now() - t0 < 8000;
    return r.fulfill({ json: { ok: true, guard: true, guard_ok: !cold, guard_state: cold ? "warming" : "ok" } });
  });
  await page.goto("/");
  await expect(page.getByTestId("guard-status")).toHaveAttribute("data-state", "warming");
  await expect(page.getByTestId("guard-status")).toContainText("waking up");
  await expect(page.getByTestId("levels-offline-banner")).toHaveCount(0);
  await reachBoss3(page);
  const tab3 = page.getByTestId("boss-tab-3");
  await expect(tab3).toBeEnabled(); // never locked for warming
  await expect(tab3).not.toHaveAttribute("aria-label", /offline/);
  await expect(page.getByTestId("guard-waking-screen")).toContainText("BOSS WAKING UP…");
  await expect(page.getByTestId("guard-waking-screen")).toContainText("~10 s");
  await expect(page.getByTestId("level-offline")).toHaveCount(0);
  await expect(page.getByLabel("Your message")).toBeDisabled();
  seenWaking = true;
  // health is polled every 2 s; once the guard answers the boss opens without any click
  await expect(page.getByTestId("guard-waking-screen")).toHaveCount(0, { timeout: 20_000 });
  expect(healthCalls).toBeGreaterThan(2);
  await expect(page.getByTestId("guard-status")).toHaveAttribute("data-state", "online");
  await expect(page.getByLabel("Your message")).toBeEnabled();
  await page.getByLabel("Your message").fill("hello boss three");
  await page.getByRole("button", { name: "FIRE" }).click();
  await expect.poll(() => st.sent.filter((m) => m.level === 3).length).toBe(1);
  await expect(page.getByTestId("level-offline")).toHaveCount(0);
});

test("guard down: offline panel with RETRY, then it recovers", async ({ page }) => {
  await mockApi(page, { wins: { 1: () => true, 2: () => true } });
  let state: "down" | "ok" = "down";
  await page.route(/\/api\/health$/, (r) =>
    r.fulfill({ json: { ok: true, guard: true, guard_ok: state === "ok", guard_state: state } }));
  await page.goto("/");
  await expect(page.getByTestId("guard-status")).toHaveAttribute("data-state", "offline");
  await expect(page.getByTestId("levels-offline-banner")).toBeVisible();
  await reachBoss3(page);
  await expect(page.getByTestId("level-offline")).toContainText("Level 3 is offline right now");
  await expect(page.getByTestId("guard-waking-screen")).toHaveCount(0);
  await expect(page.getByLabel("Your message")).toBeDisabled();
  state = "ok";
  await page.getByTestId("guard-retry").click();
  await expect(page.getByTestId("level-offline")).toHaveCount(0);
  await expect(page.getByTestId("guard-status")).toHaveAttribute("data-state", "online");
  await expect(page.getByLabel("Your message")).toBeEnabled();
});

test("an older gateway without guard_state: guard_ok false means warming, not offline", async ({ page }) => {
  await mockApi(page);
  await page.route(/\/api\/health$/, (r) => r.fulfill({ json: { ok: true, guard: true, guard_ok: false } }));
  await page.goto("/");
  await expect(page.getByTestId("guard-status")).toHaveAttribute("data-state", "warming");
  await expect(page.getByTestId("levels-offline-banner")).toHaveCount(0);
});

test("hovering or focusing a guard boss pre-warms the guard; the home page warms on load", async ({ page }) => {
  const st = await mockApi(page);
  await page.goto("/");
  await expect.poll(() => st.calls.warm ?? 0).toBeGreaterThanOrEqual(1);
  const before = st.calls.warm ?? 0;
  await page.getByTestId("boss-3").hover();
  await expect.poll(() => st.calls.warm ?? 0).toBeGreaterThan(before);
});

test("a gateway that has not answered yet (its own cold start) is warming, not offline", async ({ page }) => {
  await mockApi(page);
  const t0 = Date.now();
  await page.route(/\/api\/health$/, (r) =>
    Date.now() - t0 < 5000
      ? r.abort("failed")
      : r.fulfill({ json: { ok: true, guard: true, guard_ok: true, guard_state: "ok" } }));
  await page.goto("/");
  await expect(page.getByTestId("guard-status")).toHaveAttribute("data-state", "warming");
  await expect(page.getByTestId("levels-offline-banner")).toHaveCount(0);
  await expect(page.getByTestId("guard-status")).toHaveAttribute("data-state", "online", { timeout: 15_000 });
});
