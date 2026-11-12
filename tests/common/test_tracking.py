from breachlab.common.tracking import FakeTracker


def test_fake_tracker() -> None:
    t = FakeTracker()
    assert t.log_run("a", {"p": "1"}, {"m": 1.0}) == "fake-1"
    assert t.runs[0][0] == "a"
