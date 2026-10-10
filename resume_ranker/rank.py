"""Score every resume once, then re-rank cheaply under any weights and must-haves."""
from collections.abc import Iterable
from dataclasses import dataclass

from .parse import Profile, canonical_skill
from .score import WEIGHTS, Components, Embed, components, total


@dataclass
class Result:
    name: str
    score: float
    components: Components
    disqualified: list[str]  # must-have skills this resume lacks


def score_all(jd: Profile, cvs: list[Profile], embed: Embed) -> list[tuple[str, Components]]:
    return [(cv.name, components(jd, cv, embed)) for cv in cvs]


def rank(scored: list[tuple[str, Components]], weights: dict[str, float] = WEIGHTS,
         must_have: Iterable[str] = ()) -> list[Result]:
    must = [canonical_skill(s) for s in must_have]
    results = []
    for name, c in scored:
        have = {s for pair in c.matched for s in pair} | set(c.extra)
        results.append(Result(name, total(c.scores, weights), c, [s for s in must if s not in have]))
    return sorted(results, key=lambda r: (bool(r.disqualified), -r.score))


def to_rows(results: list[Result]) -> list[dict]:
    return [{
        "rank": i,
        "name": r.name,
        "score": round(r.score, 1),
        **{k: round(v, 2) for k, v in r.components.scores.items()},
        "disqualified": ", ".join(r.disqualified),
        "matched": ", ".join(jd_skill for jd_skill, _ in r.components.matched),
        "missing": ", ".join(r.components.missing),
        "extra": ", ".join(r.components.extra),
    } for i, r in enumerate(results, 1)]
