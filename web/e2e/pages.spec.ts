import { expect, test } from "@playwright/test";
import { mockApi, startRun } from "./mock";

test("under-the-hood renders the scorecard and live artifact links", async ({ page }) => {
  await page.route("**/api/health", (r) => r.fulfill({ json: { ok: true, guard: true } }));
  await page.goto("/under-the-hood");
  await expect(page.getByRole("heading", { name: /Scorecard \(v1, gold_test\)/ })).toBeVisible();
  await expect(page.getByRole("cell", { name: /%/ }).first()).toBeVisible();
  // latency: true value rounded to the nearest ms (2062.66 -> 2063, never down), n/a when not measured, plus the method note
  await expect(page.getByTestId("shipped-p95")).toHaveText("2063 ms");
  await expect(page.getByRole("row", { name: /qwen3-0\.6b-sft-dpo/ }).getByRole("cell").last()).toHaveText("n/a");
  await expect(page.getByTestId("latency-note")).toContainText("How latency was measured");
  await expect(page.getByTestId("latency-note")).toContainText("300 ms target is not met");
  await expect(page.getByTestId("l3-note")).toContainText("L3: same model, relaxed threshold");
  await expect(page.getByRole("link", { name: /Guard model \(Hugging Face\)/ })).toHaveAttribute("href", /https:\/\/huggingface\.co\/Pseudo-Harsh\/breachlab-guard-v1/);
  await expect(page.getByRole("link", { name: /Source code/ })).toHaveAttribute("href", /https:\/\/github\.com\/giri-harsh\/breachlab/);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});

test("model-card shows measured metrics with unrounded latency", async ({ page }) => {
  await page.route("**/api/health", (r) => r.fulfill({ json: { ok: true, guard: true } }));
  await page.goto("/model-card");
  const metrics = page.getByTestId("model-metrics");
  await expect(metrics).toContainText("Attack success (ASR)");
  await expect(metrics).toContainText(/\d+\.\d+%/);
  await expect(metrics).toContainText(/p95 latency/);
  await expect(page.getByRole("link", { name: /Dataset \(Hugging Face\)/ })).toHaveAttribute("href", /huggingface\.co\/datasets\/Pseudo-Harsh\/breachlab-data/);
});

test("nav shows guard offline when /api/health answers 503", async ({ page }) => {
  await page.route("**/api/health", (r) => r.fulfill({ status: 503, json: { detail: "guard down" } }));
  await page.goto("/model-card");
  await expect(page.getByTestId("guard-status")).toHaveAttribute("data-state", "offline");
  await expect(page.getByTestId("guard-status")).toHaveText("guard offline");
});

test("nav shows guard online when health is ok", async ({ page }) => {
  await page.route("**/api/health", (r) => r.fulfill({ json: { ok: true, guard: true } }));
  await page.goto("/leaderboard");
  await expect(page.getByTestId("guard-status")).toHaveAttribute("data-state", "online");
});

test("play: a 503 from /api/message shows the guard-offline state, not a raw error", async ({ page }) => {
  let hits = 0;
  await mockApi(page);
  await page.route("**/api/message", (r) => { hits += 1; return r.fulfill({ status: 503, json: { detail: "guard waking up, retry in a few seconds" } }); });
  await startRun(page, "Offline Tester");
  await page.getByLabel("Your message").fill("hello");
  await page.getByRole("button", { name: "FIRE" }).click();
  await expect(page.getByTestId("guard-offline")).toBeVisible();
  await expect(page.getByTestId("guard-offline")).toContainText("Guard offline");
  expect(hits).toBe(2); // one automatic retry
});

test("play: boss tabs stay locked without ?unlock=1 and open with it", async ({ page }) => {
  await mockApi(page);
  await startRun(page, "Lock Tester");
  await expect(page.getByTestId("boss-tab-3")).toBeDisabled();
  await page.goto("/play?unlock=1");
  await expect(page.getByTestId("boss-tab-3")).toBeEnabled();
  await expect(page.getByTestId("boss-tab-5")).toBeEnabled();
});

test("play: a vault-busy 503 is shown as busy, not as guard offline", async ({ page }) => {
  await mockApi(page);
  await page.route("**/api/message", (r) => r.fulfill({ status: 503, json: { detail: "the vault is busy, try again" } }));
  await startRun(page, "Busy Tester");
  await page.getByLabel("Your message").fill("hello");
  await page.getByRole("button", { name: "FIRE" }).click();
  await expect(page.getByTestId("vault-busy")).toBeVisible();
  await expect(page.getByTestId("guard-offline")).toHaveCount(0);
});

test("every page has one h1 and no horizontal scroll", async ({ page }) => {
  await mockApi(page);
  for (const path of ["/", "/play", "/leaderboard", "/under-the-hood", "/model-card", "/admin/review"]) {
    await page.goto(path);
    await expect(page.locator("h1")).toHaveCount(1);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), path).toBe(true);
  }
});

test("under-the-hood shows the animated request path and the version timeline", async ({ page }) => {
  await mockApi(page);
  await page.goto("/under-the-hood");
  for (const s of ["regex", "guard", "vault", "filter", "leak"]) await expect(page.getByTestId(`stage-${s}`)).toBeVisible();
  await expect(page.getByTestId("stage-guard")).toContainText("0.4%");
  await expect(page.getByTestId("stage-regex")).toContainText("59.4%");
  await expect(page.getByTestId("timeline")).toContainText("ADR-0082");
});
