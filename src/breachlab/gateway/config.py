"""Gateway settings from the environment (never logged)."""

from __future__ import annotations

import hashlib

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from breachlab.gateway.agent import parse_models


class GatewaySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    database_url: str = Field("sqlite://", validation_alias="NEON_DATABASE_URL")
    groq_api_key: str = Field("", validation_alias="GROQ_API_KEY")
    admin_password: str = Field("", validation_alias="ADMIN_PASSWORD")
    gateway_secret_key: str = Field("", validation_alias="GATEWAY_SECRET_KEY")
    guard_url: str = Field("", validation_alias="GUARD_URL")
    cors_origin: str = Field("*", validation_alias="CORS_ORIGIN")
    agent_model: str = Field("", validation_alias="AGENT_MODEL")  # optional: pinned first choice
    agent_models: str = Field(
        "openai/gpt-oss-20b,qwen/qwen3.8-27b,openai/gpt-oss-120b", validation_alias="AGENT_MODELS"
    )
    ip_salt: str = Field("", validation_alias="IP_SALT")
    guard_timeout_s: float = Field(3.0, validation_alias="GUARD_TIMEOUT_S")
    guard_cold_timeout_s: float = Field(
        20.0, validation_alias="GUARD_COLD_TIMEOUT_S"
    )  # first call after idle (ADR-0085)
    guard_cold_after_s: float = Field(120.0, validation_alias="GUARD_COLD_AFTER_S")
    guard_warm_grace_s: float = Field(60.0, validation_alias="GUARD_WARM_GRACE_S")
    guard_base_url: str = Field(
        "", validation_alias="GUARD_BASE_URL"
    )  # the L3 guard host (ADR-0086); empty = champion host
    guard_token: str = Field("", validation_alias="GUARD_TOKEN")  # bearer for a private Space
    champion_ttl_s: float = Field(30.0, validation_alias="CHAMPION_TTL_S")

    @property
    def vault_models(self) -> list[str]:
        """AGENT_MODELS in order; a non-empty AGENT_MODEL (older setting) is tried first."""
        return parse_models([self.agent_model, *parse_models(self.agent_models)])

    @property
    def secret_key(self) -> str:
        """GATEWAY_SECRET_KEY, else a domain-separated derivation from ADMIN_PASSWORD (documented in ADR)."""
        if self.gateway_secret_key:
            return self.gateway_secret_key
        return hashlib.sha256(f"breachlab-gateway-v1|{self.admin_password}".encode()).hexdigest()

    @property
    def salt(self) -> str:
        return (
            self.ip_salt
            or hashlib.sha256(f"breachlab-ip|{self.secret_key}".encode()).hexdigest()[:16]
        )
