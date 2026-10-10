"""Hybrid NER + semantic scoring of one resume against one job description."""
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .parse import Profile

WEIGHTS = {"skills": 0.4, "experience": 0.2, "role": 0.15, "education": 0.1, "overall": 0.15}
SKILL_MATCH = 0.75  # ponytail: fixed cosine threshold; tune with eval/ before making it configurable
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LEVEL_NAMES = {1: "diploma", 2: "bachelor", 3: "master", 4: "phd"}

Embed = Callable[[list[str]], np.ndarray]


@dataclass
class Components:
    scores: dict[str, float]
    matched: list[tuple[str, str]] = field(default_factory=list)  # (jd skill, resume skill)
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    evidence: dict[str, str] = field(default_factory=dict)


def load_embedder() -> Embed:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    return lambda texts: model.encode(list(texts), normalize_embeddings=True)


def _sim(embed: Embed, a: list[str], b: list[str]) -> np.ndarray:
    if not a or not b:
        return np.zeros((len(a), len(b)))
    return np.clip(embed(a) @ embed(b).T, 0.0, 1.0)


def skill_match(jd_skills: list[str], cv_skills: list[str], embed: Embed):
    sim = _sim(embed, jd_skills, cv_skills)
    matched, missing = [], []
    for i, skill in enumerate(jd_skills):
        j = int(sim[i].argmax()) if cv_skills else -1
        if j >= 0 and sim[i, j] >= SKILL_MATCH:
            matched.append((skill, cv_skills[j]))
        else:
            missing.append(skill)
    used = {cv for _, cv in matched}
    return matched, missing, [s for s in cv_skills if s not in used]


def _ratio(have, need, fmt) -> tuple[float, str]:
    if not need:
        return 1.0, "no requirement"
    if have is None:
        return 0.0, f"needs {fmt(need)}, not found"
    return min(1.0, have / need), f"needs {fmt(need)}, has {fmt(have)}"


def _role(jd: Profile, cv: Profile, embed: Embed) -> tuple[float, str]:
    if not jd.titles:
        return 1.0, "no title in JD"
    if not cv.titles:
        return 0.0, "no job titles found"
    sim = _sim(embed, jd.titles, cv.titles)
    i, j = np.unravel_index(sim.argmax(), sim.shape)
    return float(sim[i, j]), f"'{cv.titles[j]}' ~ '{jd.titles[i]}'"


def components(jd: Profile, cv: Profile, embed: Embed) -> Components:
    # ponytail: MiniLM truncates at 256 tokens, so "overall" sees each document's opening only
    overall = float(_sim(embed, [jd.text], [cv.text])[0, 0])
    matched, missing, extra = skill_match(jd.skills, cv.skills, embed)
    if jd.skills:
        skills = (len(matched) / len(jd.skills), f"{len(matched)}/{len(jd.skills)} required skills")
    else:
        skills = (overall, "no skills found in JD; using overall fit")
    parts = {
        "skills": skills,
        "experience": _ratio(cv.years, jd.years, lambda y: f"{y:.1f}y"),
        "role": _role(jd, cv, embed),
        "education": _ratio(cv.degree, jd.degree, LEVEL_NAMES.get),
        "overall": (overall, "whole-document similarity"),
    }
    return Components(
        scores={k: s for k, (s, _) in parts.items()},
        matched=matched, missing=missing, extra=extra,
        evidence={k: e for k, (_, e) in parts.items()},
    )


def total(scores: dict[str, float], weights: dict[str, float]) -> float:
    w = sum(weights.values())
    return 100 * sum(weights[k] * scores[k] for k in weights) / w if w else 0.0


def legacy_score(jd: Profile, cv: Profile) -> int:
    """The original notebook's points (minus its hardcoded company list): the eval baseline."""
    def low(p: Profile, key: str) -> list[str]:
        return [x.casefold() for x in p.ents.get(key, [])]

    return (5 * sum(s in low(jd, "SKILLS") for s in low(cv, "SKILLS"))
            + 10 * sum(d in low(jd, "DEGREE") for d in low(cv, "DEGREE"))
            + 10 * sum(w in low(jd, "JOBPOST") for w in low(cv, "WORKED AS")))
