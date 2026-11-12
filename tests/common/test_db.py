import sqlalchemy as sa
from sqlalchemy.orm import Session

from breachlab.common.db import Attempt, Base, GameSession, Player


def test_schema_roundtrip() -> None:
    engine = sa.create_engine("sqlite://")
    Base.metadata.create_all(engine)
    names = set(sa.inspect(engine).get_table_names())
    assert names == {"players", "sessions", "levels", "attempts", "labels", "model_versions"}
    with Session(engine) as s:
        p = Player(nickname="tester")
        s.add(p)
        s.flush()
        s.add(GameSession(id="s1", player_id=p.id, ip_hash="h"))
        s.add(Attempt(session_id="s1", level=1, message="hi"))
        s.commit()
        assert s.scalar(sa.select(sa.func.count()).select_from(Attempt)) == 1
