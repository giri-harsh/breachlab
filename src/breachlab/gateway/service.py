"""Game logic: sessions, levels, the guarded message pipeline, leaderboard (PRD 5, 6)."""

from __future__ import annotations

import hashlib
import hmac
import random
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from breachlab.common.db import Attempt, GameSession, Label, Level, Player
from breachlab.gateway.agent import AgentError, VaultAgent
from breachlab.gateway.codes import level_secret, secret_hash
from breachlab.gateway.guards import Guard, GuardOutcome, GuardUnavailableError
from breachlab.gateway.leak import detect
from breachlab.gateway.levels import MAIN_LEVELS, LevelsFile, load_levels
from breachlab.gateway.moderation import check_nickname
from breachlab.gateway.ratelimit import allowed, count_recent
from breachlab.gateway.serving import ServingRef, ServingState, resolve_label
from breachlab.gateway.telemetry import redact


def _dialect_insert(name: str) -> Any:
    if name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        return pg_insert
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert

    return sqlite_insert


CONSENT = "Prompts are logged and used to train the guard. Don't enter personal data."
CATEGORY_TEXT = {
    "injection": "prompt injection",
    "jailbreak": "jailbreak attempt",
    "exfiltration": "secret extraction",
    "none": "policy",
}
DISPLAY_REVIEWER = (
    "admin:display"  # a human label with this reviewer approves showing the prompt (CONTRACTS 7.5)
)
VERDICT_CATEGORIES = ("injection", "jailbreak", "exfiltration")
GUARD_LEVEL_INPUTS = (
    "champion",
    "qwen3-0.6b-zeroshot",
)  # input_guard values served by the HTTP guard


class GameError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status, self.detail = status, detail


def now_utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def canary_valid(key: str, header: str | None, now: float | None = None, window: int = 600) -> bool:
    """FR-10: header is ``<unix-ts>.<hex hmac(key, ts)>`` and fresh."""
    if not header or "." not in header:
        return False
    ts, sig = header.split(".", 1)
    try:
        t = int(ts)
    except ValueError:
        return False
    if abs((now if now is not None else time.time()) - t) > window:
        return False
    good = hmac.new(key.encode(), ts.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(good, sig)


def canary_header(key: str, now: float | None = None) -> str:
    ts = str(int(now if now is not None else time.time()))
    return f"{ts}.{hmac.new(key.encode(), ts.encode(), hashlib.sha256).hexdigest()}"


@dataclass
class GuardSet:
    """Guards per role. ``champion`` guards L4/L5, ``base`` guards L3, ``regex`` L2, ``challenger`` shadows."""

    regex: Guard
    base: Guard | None = None
    champion: Guard | None = None
    challenger: Guard | None = None
    serving: ServingState | None = None  # champion hot-swap; None = report the guard's own version


@dataclass
class GameService:
    db: Session
    guards: GuardSet
    agent: VaultAgent
    secret_key: str
    salt: str
    levels: LevelsFile = field(default_factory=load_levels)
    nickname_llm: Callable[[str], bool] | None = None
    agent_temperature: float = 0.7
    telemetry: Any = None
    clock: Callable[[], datetime] = field(default=now_utc)  # naive UTC; tests move the day

    # ---- levels and secrets
    def _cfg(self, level: int) -> Any:
        return self.levels.level(level)

    def daily_date(self) -> str:
        """Today's UTC date, ``YYYY-MM-DD`` (CONTRACTS 7.7)."""
        return self.clock().date().isoformat()

    def secret_for(self, sid: str, level: int) -> str:
        """The level's code: per session and level, except the daily boss, shared by everyone for one UTC day."""
        if self._cfg(level).daily:
            return level_secret(self.secret_key, "daily:" + self.daily_date(), level)
        return level_secret(self.secret_key, sid, level)

    # ---- sessions
    def ip_hash(self, ip: str) -> str:
        return hashlib.sha256(f"{self.salt}|{ip}".encode()).hexdigest()

    def start_session(self, nickname: str, ip: str, is_canary: bool = False) -> dict[str, Any]:
        ok, why = check_nickname(nickname, self.nickname_llm)
        if not ok:
            raise GameError(400, why)
        player = Player(nickname=nickname.strip())
        self.db.add(player)
        self.db.flush()
        sid = str(uuid.uuid4())
        self.db.add(
            GameSession(id=sid, player_id=player.id, ip_hash=self.ip_hash(ip), is_canary=is_canary)
        )
        self.db.commit()
        return {
            "session_id": sid,
            "consent": CONSENT,
            # prompts and hint texts stay server-side (hints are revealed one tier at a time, CONTRACTS 7.4)
            "levels": [
                lv.model_dump(exclude={"system_prompt", "hints"}) for lv in self.levels.levels
            ],
        }

    def _session(self, sid: str) -> GameSession:
        s = self.db.get(GameSession, sid)
        if s is None:
            raise GameError(404, "unknown session")
        return s

    def _level_row(self, sid: str, level: int) -> Level:
        """Get or create the (session, level) row; concurrent first messages cannot collide (INSERT .. DO NOTHING)."""
        row = self.db.get(Level, (sid, level))
        secret = self.secret_for(sid, level)
        if row is not None:
            if self._cfg(level).daily:
                self._roll_daily(row, secret_hash(secret, self.salt))
            return row
        values = {
            "session_id": sid,
            "level": level,
            "secret_hash": secret_hash(secret, self.salt),
            "messages_used": 0,
        }
        dialect = self.db.get_bind().dialect.name
        if dialect in ("postgresql", "sqlite"):
            ins = _dialect_insert(dialect)(Level).values(**values).on_conflict_do_nothing()
            self.db.execute(ins)
            return self.db.get_one(Level, (sid, level), populate_existing=True)
        row = Level(**values)  # pragma: no cover - other dialects are not deployed
        self.db.add(row)
        self.db.flush()
        return row

    def _roll_daily(self, row: Level, today_hash: str) -> None:
        """The daily row belongs to the day whose code it hashes; a new UTC day starts it afresh (one clear per day)."""
        if row.secret_hash == today_hash:
            return
        self.db.execute(
            sa.update(Level)
            .where(
                Level.session_id == row.session_id,
                Level.level == row.level,
                Level.secret_hash == row.secret_hash,
            )
            .values(
                secret_hash=today_hash,
                messages_used=0,
                cleared_at=None,
                points=0,
                hints_used=0,
                skipped_at=None,
            )
            .execution_options(synchronize_session=False)
        )
        self.db.refresh(row)

    # ---- status, streak, points (CONTRACTS 7.1, 7.3)
    def _rows(self, sid: str) -> dict[int, Level]:
        return {
            r.level: r for r in self.db.scalars(sa.select(Level).where(Level.session_id == sid))
        }

    @staticmethod
    def status_of(rows: dict[int, Level], level: int) -> str:
        """``locked | open | cleared | skipped``; level N is open when N == 1 or N-1 is cleared or skipped."""
        row = rows.get(level)
        if row is not None and row.cleared_at is not None:
            return "cleared"
        if row is not None and row.skipped_at is not None:
            return "skipped"
        if level == 1:
            return "open"
        prev = rows.get(level - 1)
        if prev is not None and (prev.cleared_at is not None or prev.skipped_at is not None):
            return "open"
        return "locked"

    def _playable(self, rows: dict[int, Level], level: int) -> bool:
        return self._cfg(level).daily or self.status_of(rows, level) != "locked"

    @staticmethod
    def streak_of(rows: dict[int, Level]) -> int:
        """Consecutive clears with no hint and no skip, replayed in time order over levels 1-5.

        A skip resets it at ``skipped_at``; a clear that used a hint resets it; a hint on a level not yet cleared
        resets it now (a hint on a skipped level moves ``skipped_at`` to the hint time, so the replay sees it).
        """
        events: list[
            tuple[datetime, int, int]
        ] = []  # (time, order, delta: +1 clean clear, 0 reset)
        pending_hint = False
        for lv in MAIN_LEVELS:
            r = rows.get(lv)
            if r is None:
                continue
            if r.skipped_at is not None:
                events.append((r.skipped_at, 0, 0))
            if r.cleared_at is not None:
                events.append((r.cleared_at, 1, 0 if (r.hints_used or 0) > 0 else 1))
            elif (r.hints_used or 0) > 0 and r.skipped_at is None:
                pending_hint = True
        streak = 0
        for _, _, delta in sorted(events, key=lambda e: (e[0], e[1])):
            streak = streak + 1 if delta else 0
        return 0 if pending_hint else streak

    @staticmethod
    def total_points(rows: dict[int, Level]) -> int:
        return sum(int(r.points or 0) for lv, r in rows.items() if lv in MAIN_LEVELS)

    def _hint_tiers(self, row: Level | None) -> list[int]:
        return list(range(1, int(row.hints_used or 0) + 1)) if row is not None else []

    def _claim_message(self, row: Level, canary: bool) -> None:
        """Atomically count this message: ``UPDATE .. SET messages_used = messages_used + 1 WHERE <still allowed>``.

        The UPDATE takes the row lock, so concurrent messages on one session serialize here (on Postgres the
        loser re-evaluates the guard after the winner commits); the cap and "already cleared" can never be
        overrun. Rolled back with everything else when a guard or the agent fails, so no message is burned.
        """
        stmt = (
            sa.update(Level)
            .where(Level.session_id == row.session_id, Level.level == row.level)
            .values(messages_used=Level.messages_used + 1)
            .execution_options(synchronize_session=False)
        )
        if not canary:
            cap = self.levels.cap(row.level)
            stmt = stmt.where(Level.messages_used < cap, Level.cleared_at.is_(None))
        if self.db.execute(stmt).rowcount != 1:  # type: ignore[attr-defined]
            self.db.refresh(row)
            if row.cleared_at is not None:
                raise GameError(409, "level already cleared")
            raise GameError(409, "no messages left on this level")
        self.db.refresh(row)

    def _cleared(self, sid: str) -> int:
        top = self.db.scalar(
            sa.select(sa.func.max(Level.level)).where(
                Level.session_id == sid,
                Level.cleared_at.is_not(None),
                Level.level.in_(MAIN_LEVELS),
            )
        )
        return int(top or 0)

    def state(self, sid: str) -> dict[str, Any]:
        self._session(sid)
        rows = self._rows(sid)
        daily = next((lv for lv in self.levels.levels if lv.daily), None)
        if daily is not None and daily.id in rows:
            self._roll_daily(rows[daily.id], secret_hash(self.secret_for(sid, daily.id), self.salt))
            self.db.commit()
        used = {lv: r.messages_used for lv, r in rows.items()}
        cap = self.levels.limits.max_messages_per_level
        levels = [
            {
                "id": cfg.id,
                "boss": cfg.boss,
                "status": self.status_of(rows, cfg.id),
                "messages_used": rows[cfg.id].messages_used if cfg.id in rows else 0,
                "points": int(rows[cfg.id].points or 0) if cfg.id in rows else 0,
                "hints_used": self._hint_tiers(rows.get(cfg.id)),
            }
            for cfg in self.levels.levels
            if cfg.id in MAIN_LEVELS
        ]
        out: dict[str, Any] = {
            "cleared": self._cleared(sid),
            "messages_used": used,
            "max_messages": cap,
            "consent": CONSENT,
            "levels": levels,
            "total_points": self.total_points(rows),
            "streak": self.streak_of(rows),
            "hint_cost": self.levels.scoring.hint_cost,
        }
        if daily is not None:
            drow = rows.get(daily.id)
            out["daily"] = {
                "level": daily.id,
                "boss": daily.boss,
                "date": self.daily_date(),
                "status": "cleared" if drow is not None and drow.cleared_at is not None else "open",
                "messages_used": drow.messages_used if drow is not None else 0,
                "max_messages": self.levels.cap(daily.id),
            }
        return out

    # ---- guards
    def _unavailable(self, level: int, alias: str, kind: str, warming: bool = False) -> GameError:
        """Fail closed: undo the half-written attempt (no message is burned) and tell the UI to retry.

        ``warming`` (the guard instance is probably booting, ADR-0085) answers 202 so the UI waits and retries on its
        own; a guard that is still silent after the grace window answers 503.
        """
        self.db.rollback()
        name = "guard_warming" if warming else "guard_unavailable"
        if self.telemetry is not None and hasattr(self.telemetry, name):
            getattr(self.telemetry, name)(level, alias, kind)
        if warming:
            return GameError(202, "guard waking up, retry in a few seconds")
        return GameError(503, "guard waking up, retry in a few seconds")

    def _stamp(self, out: GuardOutcome, alias: str, ref: ServingRef) -> GuardOutcome:
        """Champion outcomes carry the registry's serving version (serving_version()), not the guard's own."""
        if alias != "champion" or self.guards.serving is None:
            return out
        return replace(out, version=resolve_label(ref, out.version))

    def _input_guard(self, level: int, text: str, ref: ServingRef) -> GuardOutcome | None:
        cfg = self._cfg(level)
        g = {"none": None, "regex": self.guards.regex, "qwen3-0.6b-zeroshot": self.guards.base}.get(
            cfg.input_guard, self.guards.champion if cfg.input_guard == "champion" else None
        )
        if g is None:
            if cfg.input_guard == "none":
                return None
            raise GuardUnavailableError(f"guard for level {level} is not configured")
        alias = "base" if cfg.input_guard == "qwen3-0.6b-zeroshot" else ref.alias
        return self._stamp(g.score(text, "input", alias), alias, ref)

    # ---- the pipeline
    def send(
        self, sid: str, level: int, message: str, ip_hash_value: str | None = None
    ) -> dict[str, Any]:
        """One guarded message. DB conflicts (concurrent requests on one session) are retried once, then 429."""
        for attempt in (1, 2):
            try:
                return self._send_once(sid, level, message, ip_hash_value)
            except GameError:
                self.db.rollback()  # nothing from a refused request may be committed later
                raise
            except (sa.exc.IntegrityError, sa.exc.OperationalError):
                self.db.rollback()
                if attempt == 2:
                    raise GameError(
                        429, "another message from this session is still being processed"
                    ) from None
                time.sleep(0.05 + random.random() * 0.1)
        raise AssertionError("unreachable")  # pragma: no cover

    def _send_once(
        self, sid: str, level: int, message: str, ip_hash_value: str | None = None
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        sess = self._session(sid)
        lim = self.levels.limits
        if level not in {lv.id for lv in self.levels.levels}:
            raise GameError(400, "unknown level")
        message = message.strip()
        if not message or len(message) > lim.max_message_chars:
            raise GameError(400, f"message must be 1-{lim.max_message_chars} characters")
        rows = self._rows(sid)
        if not sess.is_canary and not self._playable(rows, level):
            raise GameError(403, "clear or skip the previous level first")
        cap = self.levels.cap(level)
        row = self._level_row(sid, level)
        if row.cleared_at is not None and not sess.is_canary:
            raise GameError(409, "level already cleared")
        if row.messages_used >= cap and not sess.is_canary:
            raise GameError(409, "no messages left on this level")
        self._claim_message(row, sess.is_canary)
        by_s, by_ip = count_recent(
            self.db, session_id=sid, ip_hash=ip_hash_value or sess.ip_hash, now=now_utc()
        )
        ok, why = allowed(
            by_s, by_ip, lim.rate_limit_per_session_per_min, lim.rate_limit_per_ip_per_min
        )
        if not ok and not sess.is_canary:
            raise GameError(429, why)

        secret = self.secret_for(sid, level)
        # One snapshot per request: a champion swap mid-request cannot mix versions or drop this request.
        ref = self.guards.serving.snapshot() if self.guards.serving is not None else ServingRef()
        attempt = Attempt(session_id=sid, level=level, message=message)
        reply_text = ""
        blocked = False
        layer = "none"
        p_attack: float | None = None
        ar_obj = None
        out_obj: dict[str, Any] | None = None
        cfg = self._cfg(level)
        try:
            gv = self._input_guard(level, message, ref)
        except GuardUnavailableError as exc:
            alias = "base" if cfg.input_guard == "qwen3-0.6b-zeroshot" else ref.alias
            raise self._unavailable(level, alias, "input", exc.warming) from None
        if gv is not None:
            attempt.guard_version, attempt.verdict = gv.version, gv.label
            attempt.category, attempt.confidence = gv.category, gv.confidence
            p_in = gv.p_unsafe if gv.p_unsafe is not None else gv.confidence
            if cfg.input_guard in GUARD_LEVEL_INPUTS:
                p_attack = p_in
            if (
                cfg.guard_threshold is not None
            ):  # per-level decision rule (L3, supervisor decision A3)
                blocked = p_in >= cfg.guard_threshold
            else:
                blocked = gv.label == "UNSAFE"
            if blocked:
                layer = "regex" if cfg.input_guard == "regex" else "input_guard"
        if (
            self.guards.challenger is not None
            and cfg.input_guard == "champion"
            and cfg.guard_threshold is None
        ):
            try:
                sh = self.guards.challenger.score(message, "input", "challenger")
                attempt.shadow_verdict, attempt.shadow_confidence = sh.label, sh.confidence
            except Exception:  # noqa: BLE001 - shadow must never affect players
                pass
        if not blocked:
            try:
                ar = self.agent.reply(level, secret, message, self.agent_temperature)
            except AgentError:
                self.db.rollback()
                if self.telemetry is not None and hasattr(self.telemetry, "agent_unavailable"):
                    self.telemetry.agent_unavailable(level)
                raise GameError(503, "the vault is busy, try again") from None
            reply_text = ar.text
            ar_obj = ar
            attempt.cost_usd = ar.cost_usd
            if cfg.output_filter and self.guards.champion is not None:
                try:
                    out = self._stamp(
                        self.guards.champion.score(reply_text, "output", ref.alias), ref.alias, ref
                    )
                    out_obj = {
                        "label": out.label,
                        "category": out.category,
                        "confidence": out.confidence,
                    }
                    if out.label == "UNSAFE":
                        blocked, attempt.category = True, out.category
                        attempt.verdict, attempt.confidence = "UNSAFE", out.confidence
                        layer = "output_filter"
                        p_attack = out.p_unsafe if out.p_unsafe is not None else out.confidence
                except GuardUnavailableError as exc:  # the reply is never shown unfiltered at L5
                    raise self._unavailable(level, ref.alias, "output", exc.warming) from None
        # A win is what the player actually sees: a reply hidden by the guard or the output filter is not a leak.
        leak = detect("" if blocked else reply_text, secret)
        attempt.agent_reply = reply_text
        attempt.leaked, attempt.matched_variant = leak.leaked, leak.matched_variant
        attempt.latency_ms = (time.perf_counter() - t0) * 1000
        self.db.add(attempt)
        self.db.flush()
        points_awarded = 0
        if leak.leaked:
            if level in MAIN_LEVELS:  # the daily boss is not scored (its board ranks by messages)
                streak_before = self.streak_of(
                    {k: v for k, v in rows.items() if k != level} | {level: row}
                )
                points_awarded = self.levels.scoring.points(
                    level, max(0, cap - row.messages_used), int(row.hints_used or 0), streak_before
                )
                row.points = points_awarded
            row.cleared_at = now_utc()
            self.db.add(
                Label(
                    attempt_id=attempt.id, label="UNSAFE", category=attempt.category, source="leak"
                )
            )
        self.db.commit()
        rows_after = self._rows(sid)
        shown = "" if blocked else reply_text
        if self.telemetry is not None:
            self.telemetry.request(
                {
                    "session_id": sid, "level": level, "message": message, "reply": shown, "secret": secret,
                    "guard_version": attempt.guard_version, "verdict": attempt.verdict, "category": attempt.category,
                    "confidence": attempt.confidence, "guard_latency_ms": gv.latency_ms if gv else None,
                    "agent": {"model": getattr(ar_obj, "model", None), "cost_usd": attempt.cost_usd} if ar_obj else None,
                    "output": out_obj, "leaked": leak.leaked, "variant": leak.matched_variant,
                    "latency_ms": attempt.latency_ms, "cost_usd": attempt.cost_usd,
                    "shadow": {"label": attempt.shadow_verdict, "confidence": attempt.shadow_confidence}
                    if attempt.shadow_verdict else None,
                    "is_canary": sess.is_canary,
                }
            )  # fmt: skip
        if not blocked:
            v_cat = "none"
        elif layer == "regex":
            v_cat = "keyword"
        else:
            v_cat = attempt.category if attempt.category in VERDICT_CATEGORIES else "none"
        return {
            "blocked": blocked,
            "block_category": CATEGORY_TEXT.get(attempt.category or "none", "policy")
            if blocked
            else None,
            "reply": shown,
            "leaked": leak.leaked,
            "level_cleared": leak.leaked,
            "messages_left": cap - row.messages_used,
            "guard_version": attempt.guard_version,
            "latency_ms": round(attempt.latency_ms, 1),
            "verdict": {
                "decision": "BLOCKED" if blocked else "PASSED",
                "p_attack": round(p_attack, 4) if p_attack is not None else None,
                "category": v_cat,
                "layer": layer,
            },
            "points_awarded": points_awarded,
            "total_points": self.total_points(rows_after),
            "streak": self.streak_of(rows_after),
        }

    # ---- hints and skips (CONTRACTS 7.4, 7.5)
    def _main_level(self, level: int) -> None:
        if level not in MAIN_LEVELS or level not in {lv.id for lv in self.levels.levels}:
            raise GameError(400, "hints and skips exist for levels 1-5 only")

    def hint(self, sid: str, level: int, tier: int) -> dict[str, Any]:
        """Reveal hint ``tier`` (1-3) of ``level``. Tiers unlock in order; re-reading a revealed tier is free.

        Revealing a new tier resets the streak; its cost (``hint_cost``) is taken from the level's points at the clear.
        """
        sess = self._session(sid)
        self._main_level(level)
        if tier not in (1, 2, 3):
            raise GameError(400, "tier must be 1-3")
        texts = self._cfg(level).hints
        if len(texts) < tier:
            raise GameError(404, "no hint for this tier")
        rows = self._rows(sid)
        if not sess.is_canary and not self._playable(rows, level):
            raise GameError(403, "clear or skip the previous level first")
        row = self._level_row(sid, level)
        used = int(row.hints_used or 0)
        cost = 0
        if tier > used:
            if tier != used + 1:
                raise GameError(409, f"reveal hint {used + 1} first")
            if row.cleared_at is not None:
                raise GameError(409, "level already cleared")
            values: dict[str, Any] = {"hints_used": tier}
            if (
                row.skipped_at is not None
            ):  # a hint after a skip resets the streak now (streak_of replays skipped_at)
                values["skipped_at"] = now_utc()
            n = self.db.execute(
                sa.update(Level)
                .where(Level.session_id == sid, Level.level == level, Level.hints_used == used)
                .values(**values)
                .execution_options(synchronize_session=False)
            ).rowcount  # type: ignore[attr-defined]
            if n != 1:
                self.db.rollback()
                raise GameError(409, "hint already being revealed, try again")
            cost = self.levels.scoring.hint_cost
        self.db.commit()
        self.db.refresh(row)
        rows = self._rows(sid)
        return {
            "level": level,
            "tier": tier,
            "text": texts[tier - 1],
            "cost": cost,
            "total_points": self.total_points(rows),
            "hints_used": self._hint_tiers(row),
            "streak": self.streak_of(rows),
        }

    def skip(self, sid: str, level: int) -> dict[str, Any]:
        """Skip ``level`` (0 points, streak 0, next level opens). Idempotent; 409 when cleared, 403 when locked."""
        sess = self._session(sid)
        self._main_level(level)
        rows = self._rows(sid)
        status = self.status_of(rows, level)
        if status == "cleared":
            raise GameError(409, "level already cleared")
        if status == "locked" and not sess.is_canary:
            raise GameError(403, "clear or skip the previous level first")
        row = self._level_row(sid, level)
        if row.skipped_at is None:
            self.db.execute(
                sa.update(Level)
                .where(Level.session_id == sid, Level.level == level, Level.skipped_at.is_(None))
                .values(skipped_at=now_utc())
                .execution_options(synchronize_session=False)
            )
        self.db.commit()
        rows = self._rows(sid)
        others = self.how_others_beat_it(
            level, exclude_sid=sid, caller_secret=self.secret_for(sid, level)
        )
        texts = self._cfg(level).hints
        return {
            "level": level,
            "status": "skipped",
            "next_level": level + 1 if level + 1 in MAIN_LEVELS else None,
            "how_others_beat_it": others,
            "fallback_hint": (texts[2] if len(texts) >= 3 else (texts[-1] if texts else None))
            if not others
            else None,
            "total_points": self.total_points(rows),
            "streak": 0,
        }

    def how_others_beat_it(
        self, level: int, *, exclude_sid: str, caller_secret: str, limit: int = 2
    ) -> list[dict[str, Any]]:
        """Up to ``limit`` winning prompts at ``level`` from non-canary sessions, each approved for display by a human
        review label (``reviewer == DISPLAY_REVIEWER``), with that session's code (and the caller's) as ``[SECRET]``."""
        found = self.db.execute(
            sa.select(Attempt.session_id, Attempt.message, Level.messages_used)
            .join(Label, Label.attempt_id == Attempt.id)
            .join(GameSession, GameSession.id == Attempt.session_id)
            .join(
                Level, sa.and_(Level.session_id == Attempt.session_id, Level.level == Attempt.level)
            )
            .where(
                Attempt.level == level,
                Attempt.leaked.is_(True),
                Attempt.session_id != exclude_sid,
                GameSession.is_canary.is_(False),
                Label.source == "human",
                Label.reviewer == DISPLAY_REVIEWER,
            )
            .order_by(Level.messages_used, Attempt.created_at.desc())
            .limit(limit)
        ).all()
        out = []
        for other_sid, msg, used in found:
            text = redact(msg, level_secret(self.secret_key, other_sid, level)) or ""
            text = redact(text, caller_secret) or ""
            out.append({"prompt": text, "messages": int(used)})
        return out

    # ---- boards
    def leaderboard(self, limit: int = 50) -> list[dict[str, Any]]:
        """CONTRACTS 7.6: bosses beaten desc, points desc, messages asc (then less time). Canary sessions excluded.

        ``messages`` is the sum of messages used on cleared levels (its v1 meaning); ``level`` (highest cleared) and
        ``seconds`` stay from v1. Players appear once they have beaten at least one boss.
        """
        found = self.db.execute(
            sa.select(GameSession.id, Player.nickname, GameSession.started_at, Level)
            .join(Player, Player.id == GameSession.player_id)
            .join(Level, Level.session_id == GameSession.id)
            .where(GameSession.is_canary.is_(False), Level.level.in_(MAIN_LEVELS))
        ).all()
        per: dict[str, tuple[str, datetime, dict[int, Level]]] = {}
        for sid, nick, started, row in found:
            per.setdefault(sid, (nick, started, {}))[2][row.level] = row
        agg: list[dict[str, Any]] = []
        for sid, (nick, started, rows) in per.items():
            cleared = [r for r in rows.values() if r.cleared_at is not None]
            if not cleared:
                continue
            statuses = [self.status_of(rows, lv) for lv in MAIN_LEVELS]
            top = max(r.level for r in cleared)
            agg.append(
                {
                    "session_id": sid,
                    "nickname": nick,
                    "beaten": len(cleared),
                    "skipped": statuses.count("skipped"),
                    "points": self.total_points(rows),
                    "messages": sum(r.messages_used for r in cleared),
                    "levels": statuses,
                    "level": top,
                    "seconds": max(
                        ((r.cleared_at or started) - started).total_seconds() for r in cleared
                    ),
                }
            )
        board = sorted(
            agg, key=lambda a: (-a["beaten"], -a["points"], a["messages"], a["seconds"])
        )[:limit]
        for rank, a in enumerate(board, 1):
            a["rank"] = rank
            a["seconds"] = round(a["seconds"], 1)
            a["guard_version"] = self.db.scalar(
                sa.select(Attempt.guard_version)
                .where(
                    Attempt.session_id == a.pop("session_id"),
                    Attempt.level == a["level"],
                    Attempt.leaked.is_(True),
                )
                .limit(1)
            )
        return board

    def hall_of_breaches(self, limit: int = 20) -> list[dict[str, Any]]:
        """Recent level-4/5 wins; prompt text is never exposed publicly."""
        rows = self.db.execute(
            sa.select(Player.nickname, Attempt.level, Attempt.guard_version, Attempt.created_at)
            .join(GameSession, GameSession.id == Attempt.session_id)
            .join(Player, Player.id == GameSession.player_id)
            .where(
                Attempt.leaked.is_(True),
                Attempt.level.in_((4, 5)),
                GameSession.is_canary.is_(False),
            )
            .order_by(Attempt.created_at.desc())
            .limit(limit)
        ).all()
        return [
            {"nickname": n, "level": lv, "guard_version": gv, "at": at.isoformat()}
            for n, lv, gv, at in rows
        ]
