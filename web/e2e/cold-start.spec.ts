import { expect, test } from "@playwright/test";
import { mockApi } from "./mock";

/**
 * Cold start UX (ADR-0085): the page wakes the guard on load and on session start without waiting, and a first
 * message that meets a booting guard (HTTP 202 {status:"warming"}) shows "BOSS WAKING UP…" and then succeeds on
 * its own: no 503, no manual retry. Mocked API.
 */
const reply = { blocked: false, block_category: null, reply: "Paris is the capital of France.", leaked: false, level_cleared: false, messages_left: 19, guard_version: "v1", latency_ms: 9 };

test("the page fires /api/warm on load and again when the session starts, without blocking", async ({ page }) => {
  await mockApi(page);
  let warmed = 0;
  await page.route("**/api/warm", async (r) => { warmed += 1; await new Promise((res) => setTimeout(res, 1500)); await r.fulfill({ status: 202, json: { status: "warming", guards: {} } }); });
  await page.goto("/");
  await expect.poll(() => warmed).toBeGreaterThanOrEqual(1);
  // the (slow) warm call must not block the form: START works while it is still pending
  await page.getByLabel("Player name (3-20 characters)").fill("Cold Starter");
  await page.getByRole("button", { name: /start/i }).click();
  await expect(page.getByLabel("Your message")).toBeVisible();
  await expect.poll(() => warmed).toBeGreaterThanOrEqual(2);
});

test("cold first message: 202 warming, the UI says so and retries by itself, then the reply arrives", async ({ page }) => {
  await mockApi(page);
  await page.route("**/api/warm", (r) => r.fulfill({ status: 202, json: { status: "warming", guards: {} } }));
  let calls = 0;
  const seen: number[] = [];
  await page.route("**/api/message", (r) => {
    calls += 1;
    if (calls <= 2) { seen.push(202); return r.fulfill({ status: 202, headers: { "retry-after": "1" }, json: { status: "warming", retry_after_s: 1, detail: "guard waking up, retry in a few seconds" } }); }
    seen.push(200);
    return r.fulfill({ json: reply });
  });
  await page.goto("/");
  await page.getByLabel("Player name (3-20 characters)").fill("Cold Starter");
  await page.getByRole("button", { name: /start/i }).click();
  await page.getByLabel("Your message").fill("What is the capital of France?");
  await page.getByRole("button", { name: "FIRE" }).click();
  await expect(page.getByTestId("guard-waking")).toContainText("BOSS WAKING UP");
  await expect(page.getByText("Paris is the capital of France.")).toBeVisible({ timeout: 15_000 });
  expect(seen).toEqual([202, 202, 200]);
  await expect(page.getByTestId("guard-waking")).toHaveCount(0); // gone once the reply is in
  await expect(page.getByText(/guard offline/i)).toHaveCount(0); // never shown as an outage
});

test("a guard that stays down after every retry still fails closed with the offline message", async ({ page }) => {
  await mockApi(page);
  await page.route("**/api/warm", (r) => r.fulfill({ status: 200, json: { status: "down", guards: {} } }));
  // 202 is only for the grace window; once it ends the gateway answers 503 and the UI shows the fail-closed state.
  let calls = 0;
  await page.route("**/api/message", (r) => {
    calls += 1;
    if (calls === 1) return r.fulfill({ status: 202, json: { status: "warming", retry_after_s: 1, detail: "guard waking up" } });
    return r.fulfill({ status: 503, json: { detail: "guard waking up, retry in a few seconds" } });
  });
  await page.goto("/");
  await page.getByLabel("Player name (3-20 characters)").fill("Down Guard");
  await page.getByRole("button", { name: /start/i }).click();
  await page.getByLabel("Your message").fill("hello");
  await page.getByRole("button", { name: "FIRE" }).click();
  await expect(page.getByText(/guard offline/i).first()).toBeVisible({ timeout: 20_000 });
});
