from breachlab.common.settings import Budget, load_budget


def test_default_budget_loads() -> None:
    b = load_budget()
    assert b.per_job_usd_max == 25
    assert b.monthly_usd_max == 150


def test_caps() -> None:
    b = Budget(per_job_usd_max=25, monthly_usd_max=150, always_warm_hours="x")
    assert b.check_job(25) and not b.check_job(25.01)
    assert b.check_month(100, 50) and not b.check_month(100, 50.01)
