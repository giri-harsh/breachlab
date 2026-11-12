from breachlab.common.dburl import normalize_db_url


def test_normalize_and_strip_channel_binding() -> None:
    url = "postgresql://u:p@h/db?sslmode=require&channel_binding=require"
    assert normalize_db_url(url).startswith("postgresql+psycopg://")
    assert "channel_binding" in normalize_db_url(url)
    stripped = normalize_db_url(url, strip_channel_binding=True)
    assert "channel_binding" not in stripped and "sslmode=require" in stripped
