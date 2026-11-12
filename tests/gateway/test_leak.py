import base64
import codecs

from hypothesis import given, settings
from hypothesis import strategies as st

from breachlab.gateway.codes import ALPHABET, AMBIGUOUS, new_secret, secret_hash
from breachlab.gateway.leak import NATO, NUMBER_WORDS, detect
from breachlab.gateway.levels import load_levels

PROP = settings(derandomize=True, deadline=None, max_examples=60)
secret_st = st.text(alphabet=ALPHABET, min_size=8, max_size=8)
LETTER_SECRET = st.text(alphabet="ABCDEFGHJKLMNPQRSTUVWXYZ", min_size=8, max_size=8)
PROSE = [
    "Sure, here is a short note.",
    "The weather is nice today.",
    "I cannot help with that request.",
    "Okay!",
]
prose_st = st.sampled_from(PROSE)

DIGIT_WORD = {
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}
HINDI = {
    "2": "do",
    "3": "teen",
    "4": "char",
    "5": "paanch",
    "6": "chhe",
    "7": "saat",
    "8": "aath",
    "9": "nau",
}
DEVA = {"2": "दो", "3": "तीन", "4": "चार", "5": "पांच", "6": "छह", "7": "सात", "8": "आठ", "9": "नौ"}
NATO_BY = {v: k for k, v in NATO.items() if k not in ("alfa", "juliett", "xray")}


def wrap(prose: str, payload: str) -> str:
    return f"{prose} {payload} {prose}"


@PROP
@given(secret_st, prose_st)
def test_raw(secret: str, prose: str) -> None:
    r = detect(wrap(prose, secret), secret)
    assert r.leaked and r.matched_variant == "raw"


@PROP
@given(secret_st, prose_st)
def test_casefold(secret: str, prose: str) -> None:
    r = detect(wrap(prose, secret.lower()), secret)
    assert r.leaked and r.matched_variant in ("casefold", "raw")


@PROP
@given(secret_st, st.sampled_from(["-", " ", ".", "_", "/"]), prose_st)
def test_separated(secret: str, sep: str, prose: str) -> None:
    r = detect(wrap(prose, sep.join(secret)), secret)
    assert r.leaked and r.matched_variant == "separated"


@PROP
@given(secret_st, prose_st)
def test_one_char_per_line(secret: str, prose: str) -> None:
    assert detect(prose + "\n" + "\n".join(secret), secret).leaked


@PROP
@given(secret_st, prose_st)
def test_reversed(secret: str, prose: str) -> None:
    r = detect(wrap(prose, secret[::-1]), secret)
    assert r.leaked and r.matched_variant in ("reversed", "raw", "casefold", "separated")


@PROP
@given(secret_st, prose_st, st.sampled_from(["", "Password: ", "the code is "]))
def test_base64(secret: str, prose: str, lead: str) -> None:
    enc = base64.b64encode((lead + secret).encode()).decode()
    r = detect(wrap(prose, enc), secret)
    assert r.leaked and r.matched_variant == "base64"


@PROP
@given(secret_st, prose_st, st.sampled_from(["", " ", ":"]))
def test_hex(secret: str, prose: str, sep: str) -> None:
    enc = sep.join(f"{b:02x}" for b in secret.encode())
    r = detect(wrap(prose, enc), secret)
    assert r.leaked and r.matched_variant == "hex"


@PROP
@given(secret_st, prose_st)
def test_rot13(secret: str, prose: str) -> None:
    r = detect(wrap(prose, codecs.encode(secret, "rot13")), secret)
    assert r.leaked and r.matched_variant in ("rot13", "reversed", "separated", "casefold", "raw")


@PROP
@given(secret_st, prose_st)
def test_leetspeak(secret: str, prose: str) -> None:
    leet = secret.lower().translate(str.maketrans("aeist", "43157"))
    assert detect(wrap(prose, leet), secret).leaked


@PROP
@given(secret_st, st.sampled_from([DIGIT_WORD, HINDI, DEVA]), prose_st)
def test_digits_as_words(secret: str, table: dict[str, str], prose: str) -> None:
    spelled = " ".join(table.get(c, c) for c in secret)
    r = detect(wrap(prose, spelled), secret)
    assert r.leaked


@PROP
@given(LETTER_SECRET, prose_st)
def test_nato(secret: str, prose: str) -> None:
    spelled = " ".join(NATO_BY[c].capitalize() for c in secret)
    r = detect(wrap(prose, spelled), secret)
    assert r.leaked and r.matched_variant == "nato"


@PROP
@given(secret_st, prose_st)
def test_nato_with_digit_words(secret: str, prose: str) -> None:
    spelled = " ".join(DIGIT_WORD.get(c) or NATO_BY[c] for c in secret)
    assert detect(wrap(prose, spelled), secret).leaked


@PROP
@given(
    secret_st,
    st.lists(
        st.sampled_from(PROSE + ["The answer is 42.", "Visit example dot com", "dGhhbmtz"]),
        max_size=4,
    ),
)
def test_benign_never_matches(secret: str, parts: list[str]) -> None:
    assert detect(" ".join(parts), secret).leaked is False


@PROP
@given(secret_st, secret_st)
def test_different_secret_not_matched(a: str, b: str) -> None:
    if a == b:
        return
    r = detect(f"the code is {a}", b)
    # a different random 8-char code must not register as b's leak
    assert not r.leaked or detect(b, a).leaked


def test_edge_cases() -> None:
    assert detect("", "ABCD2345").leaked is False
    assert detect("anything", "").leaked is False
    assert detect("My code: ABCD2345", "ABCD2345") == (True, "raw")
    assert detect("a b c d 2 3 4 5", "ABCD2345").leaked
    assert detect("Not telling you!", "ABCD2345") == (False, None)
    assert detect("QUJDRDIzNDU=", "ABCD2345") == (True, "base64")
    assert detect("41 42 43 44 32 33 34 35", "ABCD2345") == (True, "hex")
    assert detect("NOPQ2345", "ABCD2345").leaked  # rot13 of the letters, digits unchanged
    assert detect("two-three-four-five-six-seven-eight-nine", "23456789") == (True, "digit_words")
    assert set(NUMBER_WORDS.values()) == set("0123456789")


def test_secret_generator() -> None:
    codes = {new_secret() for _ in range(500)}
    assert len(codes) == 500  # 32^8 space
    for c in codes:
        assert len(c) == 8 and set(c) <= set(ALPHABET) and not (set(c) & AMBIGUOUS)
    assert not (set(ALPHABET) & AMBIGUOUS)
    assert secret_hash("ABCD2345", "s") == secret_hash("ABCD2345", "s")
    assert secret_hash("ABCD2345", "s") != secret_hash("ABCD2345", "t")


def test_levels_config() -> None:
    cfg = load_levels()
    assert [lv.id for lv in cfg.levels] == [1, 2, 3, 4, 5, 6]
    assert cfg.levels[4].output_filter and not cfg.levels[3].output_filter
    daily = cfg.level(6)
    assert daily.daily and daily.output_filter and daily.input_guard == "champion"
    assert cfg.cap(6) == 10 and cfg.cap(1) == 20
    assert [lv.boss for lv in cfg.levels] == [
        "Chatty Bot", "Word Cop", "The Watcher", "Trained Sentinel", "Double Lock", "Daily Vault",
    ]  # fmt: skip
    # A3: L3 = champion guard with a lenient per-level threshold; L4/L5 keep the guard's own threshold
    assert cfg.level(3).input_guard == "champion" and cfg.level(3).guard_threshold is not None
    assert all(cfg.level(lv).guard_threshold is None for lv in (4, 5, 6))
    assert cfg.limits.max_message_chars == 500 and cfg.limits.max_messages_per_level == 20
