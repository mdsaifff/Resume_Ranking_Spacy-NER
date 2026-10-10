from fakes import FakeEmbed

from resume_ranker.parse import Profile
from resume_ranker.rank import rank, score_all, to_rows
from resume_ranker.score import WEIGHTS

JD = Profile("jd", "jd text", skills=["python", "react", "docker"], years=2.0)
STRONG = Profile("strong.txt", "x", skills=["python", "react", "docker"], years=4.0)
MID = Profile("mid.txt", "y", skills=["python", "react"], years=2.0)
WEAK = Profile("weak.txt", "z", skills=["excel"])


def scored():
    return score_all(JD, [WEAK, MID, STRONG], FakeEmbed())


def test_rank_orders_by_total_score():
    assert [r.name for r in rank(scored())] == ["strong.txt", "mid.txt", "weak.txt"]


def test_reweighting_can_flip_order():
    skilled = Profile("skilled.txt", "a", skills=["python", "react", "docker"], years=0.5)
    senior = Profile("senior.txt", "b", skills=["python"], years=5.0)
    s = score_all(JD, [skilled, senior], FakeEmbed())
    assert [r.name for r in rank(s)] == ["skilled.txt", "senior.txt"]
    only_experience = {**dict.fromkeys(WEIGHTS, 0.0), "experience": 1.0}
    assert [r.name for r in rank(s, only_experience)] == ["senior.txt", "skilled.txt"]


def test_missing_must_have_sorts_last_but_stays_visible():
    results = rank(scored(), must_have=["Excel"])
    assert [r.name for r in results] == ["weak.txt", "strong.txt", "mid.txt"]
    assert results[1].disqualified == ["excel"]
    assert results[0].disqualified == []


def test_must_have_alias_is_canonicalized():  # Review Focus 4
    results = rank(scored(), must_have=["ReactJS"])
    assert [r.name for r in results if r.disqualified] == ["weak.txt"]


def test_to_rows_flattens_for_csv():
    rows = to_rows(rank(scored()))
    assert rows[0]["rank"] == 1 and rows[0]["name"] == "strong.txt"
    assert rows[0]["matched"] == "python, react, docker" and rows[0]["missing"] == ""
    assert rows[2]["extra"] == "excel"
    assert list(rows[0]) == ["rank", "name", "score", *WEIGHTS, "disqualified", "matched", "missing", "extra"]
