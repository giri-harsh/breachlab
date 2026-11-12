import { expect, test } from "@playwright/test";
import { CONSENT, mockApi, noScroll, startRun } from "./mock";

// UI logic with a mocked API: name -> bosses 1..5 beaten in order, a blocked message at boss 4, consent line.
test("play bosses 1-5 against a mocked gateway", async ({ page }) => {
  const st = await mockApi(page, {
    wins: Object.fromEntries([1, 2, 3, 4, 5].map((l) => [l, (m: string) => m.startsWith("try")])),
    blocks: { 4: (m) => m.includes("block") },
  });
  await startRun(page, "Phone Tester");
  await expect(page.getByText(CONSENT, { exact: true })).toBeVisible();
  for (let level = 1; level <= 5; level++) {
    await expect(page.getByTestId("boss-title")).toContainText(`BOSS ${level} ·`);
    if (level === 4) {
      await page.getByLabel("Your message").fill("block me");
      await page.getByRole("button", { name: "FIRE" }).click();
      await expect(page.getByTestId("verdict").last()).toHaveText("BLOCKED · 91% attack · injection · input guard");
    }
    await page.getByLabel("Your message").fill(`try level ${level}`);
    await page.getByRole("button", { name: "FIRE" }).click();
    const card = page.getByTestId("boss-defeated");
    await expect(card).toContainText("BOSS DEFEATED");
    if (level < 5) await card.getByRole("button", { name: /NEXT BOSS/ }).click();
    else await expect(card.getByRole("link", { name: /HIGH SCORES/ })).toBeVisible();
  }
  expect(st.sent).toHaveLength(6);
  expect(st.status).toEqual(["cleared", "cleared", "cleared", "cleared", "cleared"]);
  expect(await noScroll(page)).toBe(true);
});

// Live smoke (only when E2E_LIVE=1): the deployed site can start a session and answer a message.
test("live: session and message round trip", async ({ page }) => {
  test.skip(!process.env.E2E_LIVE, "set E2E_LIVE=1 to run against a deployment");
  await startRun(page, `E2E ${Date.now() % 100000}`);
  await page.getByLabel("Your message").fill("Hello, what can you do?");
  await page.getByRole("button", { name: "FIRE" }).click();
  await expect(page.getByTestId("chat").getByTestId("bot-reply").first()).toBeVisible({ timeout: 60_000 });
});
