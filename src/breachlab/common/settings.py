"""Budget configuration loader."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

DEFAULT_BUDGET_PATH = Path(__file__).resolve().parents[3] / "configs" / "budget.yaml"


class Budget(BaseModel):
    per_job_usd_max: float = Field(gt=0)
    monthly_usd_max: float = Field(gt=0)
    always_warm_hours: str

    def check_job(self, projected_usd: float) -> bool:
        """True if a single job's projected cost is within the per-job cap."""
        return projected_usd <= self.per_job_usd_max

    def check_month(self, spent_usd: float, projected_usd: float) -> bool:
        return spent_usd + projected_usd <= self.monthly_usd_max


def load_budget(path: Path = DEFAULT_BUDGET_PATH) -> Budget:
    with path.open(encoding="utf-8") as fh:
        return Budget.model_validate(yaml.safe_load(fh))
