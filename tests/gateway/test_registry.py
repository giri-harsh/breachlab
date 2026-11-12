import pytest

from breachlab.flywheel.registry import promote, rollback


class V:
    def __init__(self, version: str) -> None:
        self.version = version


class FakeClient:
    def __init__(self, aliases: dict[str, str]) -> None:
        self.aliases = dict(aliases)

    def get_model_version_by_alias(self, name: str, alias: str) -> V:
        if alias not in self.aliases:
            raise KeyError(alias)
        return V(self.aliases[alias])

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.aliases[alias] = version

    def delete_registered_model_alias(self, name: str, alias: str) -> None:
        self.aliases.pop(alias, None)


def test_promote_and_rollback_roundtrip() -> None:
    c = FakeClient({"champion": "1", "challenger": "2"})
    m = promote(c)
    assert (
        (m.version, m.previous) == ("2", "1")
        and c.aliases["champion"] == "2"
        and c.aliases["previous"] == "1"
    )
    r = rollback(c)
    assert (
        (r.version, r.previous) == ("1", "2")
        and c.aliases["champion"] == "1"
        and c.aliases["previous"] == "2"
    )
    assert r.seconds < 1


def test_first_promotion_and_errors() -> None:
    c = FakeClient({"challenger": "1"})
    assert promote(c).previous is None and c.aliases["champion"] == "1"
    with pytest.raises(ValueError):
        promote(c, "1")
    with pytest.raises(ValueError):
        rollback(c)
    with pytest.raises(ValueError):
        promote(FakeClient({}))
    assert promote(FakeClient({"champion": "3"}), "4").version == "4"
