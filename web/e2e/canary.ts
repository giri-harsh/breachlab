import crypto from "node:crypto";

/**
 * Test-only canary credentials (FR-10, src/breachlab/gateway/service.py `canary_header`).
 * The gateway key is GATEWAY_SECRET_KEY, falling back to sha256("breachlab-gateway-v1|" + ADMIN_PASSWORD)
 * (src/breachlab/gateway/config.py). Header = "<unix-ts>.<hex hmac-sha256(key, ts)>", valid for 10 minutes.
 */
export function gatewayKey(env: NodeJS.ProcessEnv = process.env): string | null {
  if (env.GATEWAY_SECRET_KEY) return env.GATEWAY_SECRET_KEY;
  if (env.ADMIN_PASSWORD) return crypto.createHash("sha256").update(`breachlab-gateway-v1|${env.ADMIN_PASSWORD}`).digest("hex");
  return null;
}

export function canaryHeader(key: string, nowMs: number = Date.now()): string {
  const ts = String(Math.floor(nowMs / 1000));
  return `${ts}.${crypto.createHmac("sha256", key).update(ts).digest("hex")}`;
}
