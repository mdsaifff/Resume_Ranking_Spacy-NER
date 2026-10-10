"""Text extraction, NER, and normalization of resumes and job descriptions into Profiles."""
import json
import os
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import pymupdf
import spacy
from spacy.matcher import PhraseMatcher

PKG = Path(__file__).parent
MODELS_DIR = Path(os.environ.get("RESUME_RANKER_MODELS", PKG.parent / "models"))


@dataclass
class Profile:
    name: str
    text: str
    skills: list[str] = field(default_factory=list)  # canonical, deduped
    titles: list[str] = field(default_factory=list)
    years: float | None = None  # resume: total experience; JD: required
    degree: int | None = None  # 1 diploma, 2 bachelor, 3 master, 4 phd
    ents: dict[str, list[str]] = field(default_factory=dict)  # raw NER output


def read_text(data: bytes, filename: str) -> str:
    """Text of a .pdf or .txt file. ValueError if the type is unsupported or there is no text."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            text = "".join(page.get_text() for page in pdf)
    elif ext == ".txt":
        text = data.decode("utf-8-sig", errors="replace")
    else:
        raise ValueError(f"unsupported file type '{ext or 'none'}' (use .pdf or .txt)")
    if not text.strip():
        raise ValueError("no extractable text (scanned PDF?)")
    return text


def read_all(items: Iterable[tuple[str, bytes]]) -> tuple[list[tuple[str, str]], list[str]]:
    """Read every (filename, bytes); bad files become warnings instead of errors."""
    docs, warnings = [], []
    for name, data in items:
        try:
            docs.append((name, read_text(data, name)))
        except Exception as e:  # corrupt PDFs raise various pymupdf errors
            warnings.append(f"skipped {name}: {e}")
    return docs, warnings


def _clean(s: str) -> str:
    s = " ".join(s.casefold().split())
    return s.lstrip(",;:|-•* ").rstrip(".,;:|-•* ")  # keep leading "." for ".net"


@cache
def _aliases() -> dict[str, str]:
    raw = json.loads((PKG / "skills.json").read_text(encoding="utf-8"))
    return {_clean(a): canon for canon, aliases in raw.items() for a in [canon, *aliases]}


def canonical_skill(s: str) -> str:
    s = _clean(s)
    return _aliases().get(s, s)


def dictionary_skills(nlp: spacy.Language, text: str) -> list[str]:
    """Known skills found verbatim in text; backs up the weak JD NER model (F1 0.58)."""
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    matcher.add("SKILL", [nlp.make_doc(a) for a in _aliases()])
    doc = nlp.make_doc(text)
    return _dedupe(canonical_skill(doc[s:e].text) for _, s, e in matcher(doc))


_YEARS = re.compile(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b", re.I)
_MONTHS = re.compile(r"(\d+)\s*months?\b", re.I)
_REQUIRED = re.compile(r"(\d+(?:\.\d+)?)\s*(?:[-–<+]|to)?\s*(?:\d+)?\s*\+?\s*(?:years?|yrs?)", re.I)


def duration_years(s: str) -> float:
    """'3 years 9 months' -> 3.75; unparseable -> 0."""
    y, m = _YEARS.search(s), _MONTHS.search(s)
    return (float(y[1]) if y else 0.0) + (int(m[1]) / 12 if m else 0.0)


def required_years(s: str) -> float | None:
    """Lower bound of a JD requirement: '3-5 years' -> 3, '2+ years' -> 2; None if no year count."""
    m = _REQUIRED.search(s)
    return float(m[1]) if m else None


_DEGREE_LEVELS = [  # highest first; matched against casefolded text with . ' ’ removed
    (4, r"\b(phd|doctor|doctorate)\b"),
    (3, r"\b(masters?|ms|msc|mtech|mba|mca|postgraduate|post graduate)\b"),
    (2, r"\b(bachelors?|undergraduate|bs|bsc|be|btech|bca|bcom|bse|ba|bba|degree)\b"),
    (1, r"(\+2|\b(associates?|diploma|high school)\b)"),
]


def degree_levels(s: str) -> list[int]:
    s = re.sub(r"[.'’]", "", s.casefold())
    return [level for level, pattern in _DEGREE_LEVELS if re.search(pattern, s)]


def _dedupe(items) -> list[str]:
    return [s for s in dict.fromkeys(items) if s]


def resume_profile(name: str, text: str, ents: dict[str, list[str]]) -> Profile:
    durations = ents.get("YEARS OF EXPERIENCE", [])
    levels = [lvl for d in ents.get("DEGREE", []) for lvl in degree_levels(d)]
    return Profile(
        name, text,
        skills=_dedupe(canonical_skill(s) for s in ents.get("SKILLS", [])),
        titles=_dedupe(ents.get("WORKED AS", []) + ents.get("DESIGNATION", [])),
        # ponytail: overlapping jobs double-count; merge date ranges if that skews rankings
        years=sum(map(duration_years, durations)) if durations else None,
        degree=max(levels, default=None),
        ents=ents,
    )


def jd_profile(name: str, text: str, ents: dict[str, list[str]], dict_skills: list[str]) -> Profile:
    reqs = [y for e in ents.get("EXPERIENCE", []) if (y := required_years(e)) is not None]
    levels = [lvl for d in ents.get("DEGREE", []) for lvl in degree_levels(d)]
    return Profile(
        name, text,
        skills=_dedupe([canonical_skill(s) for s in ents.get("SKILLS", [])] + dict_skills),
        titles=_dedupe(ents.get("JOBPOST", [])),
        years=max(reqs, default=None),  # strictest stated requirement
        degree=min(levels, default=None),  # lowest acceptable degree
        ents=ents,
    )


def load_nlp(kind: str) -> spacy.Language:
    path = MODELS_DIR / kind / "model-best"
    if not path.exists() and (repo := os.environ.get("RESUME_RANKER_HF_REPO")):
        from huggingface_hub import snapshot_download
        snapshot_download(repo, allow_patterns=[f"{kind}/model-best/*"], local_dir=MODELS_DIR)
    if not path.exists():
        raise FileNotFoundError(
            f"No {kind} model at {path}. Run: python -m resume_ranker train {kind} "
            "(or set RESUME_RANKER_HF_REPO to download it)"
        )
    return spacy.load(path)


def entities(nlp: spacy.Language, text: str) -> dict[str, list[str]]:
    ents = defaultdict(list)
    for ent in nlp(text).ents:
        ents[ent.label_].append(" ".join(ent.text.split()))
    return dict(ents)


def parse_resume(nlp: spacy.Language, name: str, text: str) -> Profile:
    return resume_profile(name, text, entities(nlp, text))


def parse_jd(nlp: spacy.Language, name: str, text: str) -> Profile:
    return jd_profile(name, text, entities(nlp, text), dictionary_skills(nlp, text))
