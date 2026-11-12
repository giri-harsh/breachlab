"""YAML-backed pydantic-settings configs for training and serving (env overrides allowed)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

CONFIG_DIR = Path(
    os.environ.get("BREACHLAB_CONFIG_DIR") or Path(__file__).resolve().parents[3] / "configs"
)


class LoraConfig(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


class SftConfig(BaseModel):
    epochs: int = 3
    lr: float = 2e-4
    scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    effective_batch: int = 32
    per_device_batch: int = 4
    early_stop_patience: int = 1
    precision: Literal["bf16", "fp16"] = "fp16"
    lora: LoraConfig = Field(default_factory=LoraConfig)

    @property
    def grad_accum(self) -> int:
        return max(1, self.effective_batch // self.per_device_batch)


class MiningConfig(BaseModel):
    pool_size: int = 5000
    near_miss_margin: float = 0.2


class DpoConfig(BaseModel):
    beta: float = 0.1
    epochs: int = 1
    lr: float = 5e-6
    per_device_batch: int = 4
    grad_accum: int = 8


class CalibrationConfig(BaseModel):
    ece_bins: int = 15
    max_fpr: float = 0.02


class BudgetConfig(BaseModel):
    gpu_usd_per_hour: float = 0.0
    per_job_usd_max: float = 25.0


class TrainConfig(BaseSettings):
    model_config = SettingsConfigDict(
        yaml_file=CONFIG_DIR / "train.yaml",
        env_prefix="BREACHLAB_TRAIN_",
        env_nested_delimiter="__",
    )
    base_model: str = "Qwen/Qwen3-0.6B"
    seed: int = 0
    data_version: str = "v1"
    data_dir: Path = Path("data/processed/v1")
    output_dir: Path = Path("out")
    max_len: int = 512
    sft: SftConfig = Field(default_factory=SftConfig)
    mining: MiningConfig = Field(default_factory=MiningConfig)
    dpo: DpoConfig = Field(default_factory=DpoConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    mlflow_experiment: str = "breachlab-train"
    hf_namespace: str = "Pseudo-Harsh"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings, env_settings, YamlConfigSettingsSource(settings_cls))

    @property
    def short_name(self) -> str:
        return self.base_model.split("/")[-1].lower().replace("qwen3-", "")


class Verbalizer(BaseModel):
    word: str
    form: str
    id: int


class MaxTokens(BaseModel):
    input_guard: int = 512
    output_filter: int = 384


class ServeConfig(BaseSettings):
    model_config = SettingsConfigDict(
        yaml_file=CONFIG_DIR / "serve.yaml",
        env_prefix="BREACHLAB_SERVE_",
        env_nested_delimiter="__",
    )
    models: list[str] = Field(default_factory=list)
    verbalizers: dict[str, Verbalizer]
    identical_across_models: bool = True
    max_tokens: MaxTokens = Field(default_factory=MaxTokens)
    threshold: float | None = None
    temperature: float | None = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings, env_settings, YamlConfigSettingsSource(settings_cls))

    LABELS: tuple[str, ...] = ("safe", "injection", "jailbreak", "exfiltration")

    def token_ids(self) -> list[int]:
        """Verbalizer token ids in the fixed label order (safe, injection, jailbreak, exfiltration)."""
        return [self.verbalizers[k].id for k in self.LABELS]

    def words(self) -> list[str]:
        return [self.verbalizers[k].word for k in self.LABELS]
