# Resume Ranker v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Colab-only resume scorer into a runnable package that ranks many resumes against a job description with an explainable hybrid NER + semantic score. It ships a CLI, a Streamlit app and an evaluation harness.

**Architecture:** `resume_ranker/parse.py` turns files into text, runs the custom spaCy NER models, and normalizes entities into a `Profile`. `score.py` computes five [0,1] components per (JD, resume) pair plus evidence. `rank.py` applies weights and must-haves cheaply on top of cached components. `cli.py`, `app.py` and `eval/evaluate.py` are thin front-ends over those three modules.

**Tech Stack:** Python 3.11, spaCy 3.8 (custom NER, retrained on CPU via tok2vec), PyMuPDF, sentence-transformers (`all-MiniLM-L6-v2`), Streamlit ≥1.40, pandas, pytest, huggingface_hub (optional model hosting).

**Spec:** `docs/superpowers/specs/2026-10-10-resume-ranker-v2-design.md`

## Global Constraints

- Must work end-to-end on CPU only (no GPU on the dev machine); `--gpu` training is optional.
- Commands below use the Windows venv path `.venv/Scripts/python`; on Linux/macOS use `.venv/bin/python`.
- Inputs: `.pdf` and `.txt` only. Anything else is skipped with a warning, never a crash.
- Component scores are in [0,1]; total score is `100 * Σ wᵢ·sᵢ / Σ wᵢ` on 0–100.
- Default weights: skills .4, experience .2, role .15, education .1, overall .15. Skill match threshold: cosine ≥ 0.75. Embedding model: `sentence-transformers/all-MiniLM-L6-v2`.
- No company-prestige signal.
- Disqualified candidates (missing a must-have) stay visible and sort last, with the reason.
- `models/` is never committed (gitignored); trained models come from `python -m resume_ranker train` or the HF Hub repo named by env `RESUME_RANKER_HF_REPO`.
- All file I/O is UTF-8. CLI output must not crash on non-ASCII text in a Windows cp1252 console.
- Weights/thresholds are module constants; no config files.

## Review Focus

1. All weight sliders at 0 → every total is 0, nothing divides by zero, the table still renders (test in Task 4).
2. A JD where nothing is extracted as a skill → the skills component falls back to overall fit and says so, with no crash on empty similarity matrices (test in Task 4).
3. A resume with no entities at all (bad PDF layout) → components score 0 with readable evidence, and the run doesn't crash (test in Task 4).
4. Must-have typed as an alias or in different case ("ReactJS" vs JD's "React") → canonicalized before comparison, so the candidate isn't wrongly disqualified (test in Task 5).
5. Uploading a corrupt PDF, an empty file, or a `.docx` alongside good resumes → those are skipped with a warning naming the file, and the rest are ranked (test in Task 1).

## File Map

| File | Responsibility |
|---|---|
| `resume_ranker/parse.py` | bytes→text, model loading, NER→entities, skill/years/degree normalization, `Profile` |
| `resume_ranker/skills.json` | canonical skill → aliases |
| `resume_ranker/score.py` | embedder, per-pair components + evidence, weighted total, legacy baseline |
| `resume_ranker/rank.py` | score once, rank under weights/must-haves, flatten to rows |
| `resume_ranker/cli.py`, `__main__.py` | `rank` and `train` commands |
| `app.py` | Streamlit UI |
| `eval/evaluate.py` | `list` / `pool` / `score` subcommands, NDCG@5 and P@5 |
| `configs/cpu.cfg`, `configs/gpu.cfg` | spaCy training configs |
| `data/{resume,jd}/{train,test}.spacy` | training data (moved from `Model/`) |
| `samples/` | fictional JD + 4 resumes for demos |
| `tests/` | `fakes.py`, `test_parse.py`, `test_score.py`, `test_rank.py`, `test_evaluate.py` |

---

### Task 1: Project setup and file reading

**Files:**
- Create: `requirements.txt`, `.gitignore`, `pyproject.toml`, `resume_ranker/__init__.py`, `resume_ranker/parse.py`, `tests/test_parse.py`
- Move: `Model/*/{train,test}_data.spacy` → `data/{resume,jd}/{train,test}.spacy`; `resume_ranker(spacy).py` → `notebooks/resume_ranker_colab.py`
- Delete: `Model/` (stale configs without weights; old F1 numbers are recorded in Task 9's README)

**Interfaces:**
- Produces: `parse.read_text(data: bytes, filename: str) -> str` (raises `ValueError` on unsupported type or no text); `parse.read_all(items: Iterable[tuple[str, bytes]]) -> tuple[list[tuple[str, str]], list[str]]` returning `(name, text)` docs and warning strings.

- [ ] **Step 1: Restructure the repo**

```bash
cd "D:/Git Demo/Resume_Ranking_Spacy-NER"
mkdir -p data/resume data/jd notebooks resume_ranker tests
git mv Model/Resume/train_data.spacy data/resume/train.spacy
git mv Model/Resume/test_data.spacy data/resume/test.spacy
git mv "Model/Job Description/train_data.spacy" data/jd/train.spacy
git mv "Model/Job Description/test_data.spacy" data/jd/test.spacy
git mv "resume_ranker(spacy).py" notebooks/resume_ranker_colab.py
git rm -r -q Model
```

- [ ] **Step 2: Write `requirements.txt`, `.gitignore`, `pyproject.toml`, empty `resume_ranker/__init__.py`**

`requirements.txt`:
```
spacy>=3.8,<3.9
pymupdf>=1.24.3
sentence-transformers>=3.0
huggingface_hub>=0.24
streamlit>=1.40
pandas>=2.0
pytest>=8.0
```

`.gitignore`:
```
.venv/
__pycache__/
models/
eval/pool/
*.csv
```

`pyproject.toml`:
```toml
[tool.pytest.ini_options]
pythonpath = [".", "eval"]
testpaths = ["tests"]
```

- [ ] **Step 3: Create the venv and install**

Run: `python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt`
Expected: installs without errors (torch CPU wheel pulled in by sentence-transformers; a few minutes).

- [ ] **Step 4: Write the failing tests** in `tests/test_parse.py`

```python
import pymupdf
import pytest

from resume_ranker.parse import read_all, read_text


def test_read_text_txt_handles_utf8_and_bom():
    assert read_text("\ufeffhéllo · world".encode("utf-8"), "a.txt") == "héllo · world"


def test_read_text_pdf():
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "Python developer")
    assert "Python developer" in read_text(pdf.tobytes(), "cv.PDF")


def test_read_text_pdf_without_text_layer():
    pdf = pymupdf.open()
    pdf.new_page()
    with pytest.raises(ValueError, match="no extractable text"):
        read_text(pdf.tobytes(), "scan.pdf")


@pytest.mark.parametrize("data,name", [(b"   \n", "a.txt"), (b"x", "a.docx"), (b"x", "noext")])
def test_read_text_rejects(data, name):
    with pytest.raises(ValueError):
        read_text(data, name)


def test_read_all_skips_bad_files():  # Review Focus 5
    docs, warnings = read_all([
        ("good.txt", b"python"), ("empty.txt", b""), ("broken.pdf", b"not a pdf"), ("cv.docx", b"x"),
    ])
    assert docs == [("good.txt", "python")]
    assert [w.split(":")[0] for w in warnings] == ["skipped empty.txt", "skipped broken.pdf", "skipped cv.docx"]
```

- [ ] **Step 5: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_parse.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'resume_ranker.parse'` (or ImportError).

- [ ] **Step 6: Implement** `resume_ranker/parse.py`

```python
"""Text extraction, NER, and normalization of resumes and job descriptions into Profiles."""
from collections.abc import Iterable
from pathlib import Path

import pymupdf


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
```

- [ ] **Step 7: Run tests**

Run: `.venv/Scripts/python -m pytest tests/test_parse.py -v`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "Restructure repo into package; add file reading with bad-file warnings"
```

---

### Task 2: Entity normalization and profiles

**Files:**
- Create: `resume_ranker/skills.json`
- Modify: `resume_ranker/parse.py` (append)
- Test: `tests/test_parse.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `Profile` dataclass: `name: str, text: str, skills: list[str] = [], titles: list[str] = [], years: float | None = None, degree: int | None = None, ents: dict[str, list[str]] = {}`
  - `canonical_skill(s: str) -> str`
  - `dictionary_skills(nlp: spacy.Language, text: str) -> list[str]` (canonical names)
  - `duration_years(s: str) -> float`, `required_years(s: str) -> float | None`, `degree_levels(s: str) -> list[int]` (1 diploma, 2 bachelor, 3 master, 4 phd; highest first)
  - `resume_profile(name, text, ents) -> Profile` (years = sum of job durations, degree = highest)
  - `jd_profile(name, text, ents, dict_skills: list[str]) -> Profile` (years = largest stated requirement, degree = lowest acceptable)

- [ ] **Step 1: Write `resume_ranker/skills.json`**

Canonical names are keys. Ambiguous English words (go, rest, excel, express, cv, node) are deliberately left out, because the dictionary is scanned over raw JD text.
```json
{
  "python": ["python3", "python 3"],
  "java": [],
  "javascript": ["js", "es6", "ecmascript"],
  "typescript": ["ts"],
  "c++": ["cpp"],
  "c#": ["csharp", "c sharp"],
  ".net": ["dotnet", "asp.net", ".net core", "asp.net core"],
  "golang": [],
  "rust": [],
  "php": [],
  "ruby": ["ruby on rails", "rails"],
  "kotlin": [],
  "swift": [],
  "sql": ["t-sql", "pl/sql"],
  "mysql": [],
  "postgresql": ["postgres"],
  "mongodb": ["mongo"],
  "redis": [],
  "react": ["reactjs", "react.js"],
  "angular": ["angularjs", "angular.js"],
  "vue": ["vuejs", "vue.js"],
  "node.js": ["nodejs"],
  "express.js": ["expressjs"],
  "django": [],
  "flask": [],
  "fastapi": [],
  "spring": ["spring boot", "springboot"],
  "html": ["html5"],
  "css": ["css3"],
  "rest api": ["restful", "rest apis", "restful api", "restful apis", "rest services"],
  "graphql": [],
  "git": ["github", "gitlab"],
  "docker": [],
  "kubernetes": ["k8s"],
  "aws": ["amazon web services"],
  "azure": ["microsoft azure"],
  "gcp": ["google cloud", "google cloud platform"],
  "linux": [],
  "ci/cd": ["cicd", "ci cd"],
  "jenkins": [],
  "celery": [],
  "machine learning": ["ml"],
  "deep learning": ["dl"],
  "nlp": ["natural language processing"],
  "computer vision": [],
  "tensorflow": [],
  "pytorch": ["torch"],
  "scikit-learn": ["sklearn", "scikit learn"],
  "pandas": [],
  "numpy": [],
  "spark": ["apache spark", "pyspark"],
  "data analysis": ["data analytics"],
  "tableau": [],
  "power bi": ["powerbi"],
  "agile": ["scrum"],
  "jira": [],
  "solidity": [],
  "blockchain": []
}
```

- [ ] **Step 2: Write failing tests** (append to `tests/test_parse.py`; add `import spacy` and extend the import line)

```python
import spacy

from resume_ranker.parse import (
    canonical_skill, degree_levels, dictionary_skills, duration_years, jd_profile,
    required_years, resume_profile,
)


@pytest.mark.parametrize("raw,canon", [
    ("ReactJS", "react"), (" Python3 ", "python"), ("Node.js.", "node.js"), (".NET", ".net"),
    ("C#", "c#"), ("- Docker", "docker"), ("Underwater  Basket Weaving", "underwater basket weaving"),
])
def test_canonical_skill(raw, canon):
    assert canonical_skill(raw) == canon


def test_dictionary_skills_finds_aliases_case_insensitively():
    found = dictionary_skills(spacy.blank("en"), "We need ReactJS, PYTHON and Docker; go-getters welcome.")
    assert found == ["react", "python", "docker"]


@pytest.mark.parametrize("s,years", [
    ("3 years 9 months", 3.75), ("1 year 11 months", 1 + 11 / 12), ("7 months", 7 / 12),
    ("2 yrs", 2), ("5 meses", 0),
])
def test_duration_years(s, years):
    assert duration_years(s) == pytest.approx(years)


@pytest.mark.parametrize("s,need", [
    ("3-5 years", 3), ("2+ years", 2), ("5<10 years", 5), ("3 yrs+", 3), ("2 -5 years", 2),
    ("Minimum 4 years\ufffd experience", 4), ("7+ years of experience with PHP", 7),
    ("3 to 5 years", 3), ("strong communication", None),
])
def test_required_years(s, need):
    assert required_years(s) == need


@pytest.mark.parametrize("s,levels", [
    ("Bachelor's degree,\xa0Computer Science", [2]),
    ("Doctor of Philosophy (Ph.D.),\xa0Watershed Management", [4]),
    ("M.S. or Ph.D. in engineering", [4, 3]),
    ("Bachelor\ufffds/Master\ufffds degree in Engineering", [3, 2]),
    ("M.Tech/B.tech/B.E/BCA in Computer Science", [3, 2]),
    ("High School Diploma,\xa0Physical Sciences", [1]),
    ("+2 science , Mathematics", [1]),
    ("Finance", []),
])
def test_degree_levels(s, levels):
    assert degree_levels(s) == levels


def test_resume_profile_sums_durations_and_takes_highest_degree():
    p = resume_profile("cv.txt", "text", {
        "SKILLS": ["Python", "python3", "ReactJS"],
        "WORKED AS": ["Data Scientist", "Data Scientist"],
        "YEARS OF EXPERIENCE": ["2 years 6 months", "1 year"],
        "DEGREE": ["Bachelor's degree, CS", "Master's degree, Statistics"],
    })
    assert p.skills == ["python", "react"]
    assert p.titles == ["Data Scientist"]
    assert p.years == pytest.approx(3.5)
    assert p.degree == 3


def test_resume_profile_without_entities():
    p = resume_profile("cv.txt", "text", {})
    assert (p.skills, p.titles, p.years, p.degree) == ([], [], None, None)


def test_jd_profile_merges_dictionary_skills_and_reads_requirements():
    p = jd_profile("jd.txt", "text", {
        "SKILLS": ["Python", "Docker."],
        "JOBPOST": ["Backend Engineer"],
        "EXPERIENCE": ["3-5 years", "2+ years of Cloud"],
        "DEGREE": ["Bachelor's or Master's in CS"],
    }, dict_skills=["docker", "kubernetes"])
    assert p.skills == ["python", "docker", "kubernetes"]
    assert p.titles == ["Backend Engineer"]
    assert p.years == 3
    assert p.degree == 2
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_parse.py -v`
Expected: FAIL with ImportError on `canonical_skill`.

- [ ] **Step 4: Implement** (add to `resume_ranker/parse.py`; merge the new imports into the top of the file)

```python
import json
import re
from dataclasses import dataclass, field
from functools import cache

import spacy
from spacy.matcher import PhraseMatcher

PKG = Path(__file__).parent


@dataclass
class Profile:
    name: str
    text: str
    skills: list[str] = field(default_factory=list)  # canonical, deduped
    titles: list[str] = field(default_factory=list)
    years: float | None = None  # resume: total experience; JD: required
    degree: int | None = None  # 1 diploma, 2 bachelor, 3 master, 4 phd
    ents: dict[str, list[str]] = field(default_factory=dict)  # raw NER output


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
```

- [ ] **Step 5: Run tests**

Run: `.venv/Scripts/python -m pytest tests/test_parse.py -v`
Expected: all PASS. If a regex case fails, fix the regex; don't change the test's expected value. The test strings were sampled from the real training data.

- [ ] **Step 6: Commit**

```bash
git add resume_ranker tests
git commit -m "Normalize skills, experience and degrees into Profiles"
```

---

### Task 3: Training command and model loading

**Files:**
- Create: `configs/cpu.cfg`, `configs/gpu.cfg` (generated), `resume_ranker/cli.py`, `resume_ranker/__main__.py`
- Modify: `resume_ranker/parse.py` (append)
- Test: `tests/test_parse.py` (append)

**Interfaces:**
- Consumes: `resume_profile`, `jd_profile`, `dictionary_skills` (Task 2).
- Produces:
  - `parse.MODELS_DIR: Path` (env `RESUME_RANKER_MODELS`, default `<repo>/models`)
  - `parse.load_nlp(kind: str) -> spacy.Language` for `kind in {"resume", "jd"}` (loads `MODELS_DIR/kind/model-best`; downloads from HF if `RESUME_RANKER_HF_REPO` set; else `FileNotFoundError` naming the train command)
  - `parse.entities(nlp, text) -> dict[str, list[str]]`
  - `parse.parse_resume(nlp, name, text) -> Profile`, `parse.parse_jd(nlp, name, text) -> Profile`
  - `cli.main(argv: list[str] | None = None)` with subcommand `train {resume,jd} [--gpu]`

- [ ] **Step 1: Write failing tests** (append to `tests/test_parse.py`)

```python
from resume_ranker import parse
from resume_ranker.parse import entities, parse_jd


def test_entities_and_parse_jd_with_rule_based_nlp():
    nlp = spacy.blank("en")
    nlp.add_pipe("entity_ruler").add_patterns([
        {"label": "SKILLS", "pattern": "Python"},
        {"label": "JOBPOST", "pattern": "Data Engineer"},
        {"label": "EXPERIENCE", "pattern": "3+ years"},
    ])
    text = "Data Engineer with 3+ years of Python and Docker"
    assert entities(nlp, text) == {"JOBPOST": ["Data Engineer"], "EXPERIENCE": ["3+ years"], "SKILLS": ["Python"]}
    jd = parse_jd(nlp, "jd.txt", text)
    assert (jd.skills, jd.titles, jd.years) == (["python", "docker"], ["Data Engineer"], 3)


def test_load_nlp_explains_how_to_get_missing_model(tmp_path, monkeypatch):
    monkeypatch.setattr(parse, "MODELS_DIR", tmp_path)
    monkeypatch.delenv("RESUME_RANKER_HF_REPO", raising=False)
    with pytest.raises(FileNotFoundError, match="python -m resume_ranker train resume"):
        parse.load_nlp("resume")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_parse.py -v`
Expected: FAIL with ImportError on `entities`.

- [ ] **Step 3: Implement** (append to `resume_ranker/parse.py`; add `import os` and `from collections import defaultdict` at the top)

```python
MODELS_DIR = Path(os.environ.get("RESUME_RANKER_MODELS", PKG.parent / "models"))


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
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python -m pytest tests/test_parse.py -v`
Expected: all PASS.

- [ ] **Step 5: Generate training configs**

```bash
mkdir -p configs
.venv/Scripts/python -m spacy init config configs/cpu.cfg --lang en --pipeline ner --optimize efficiency
.venv/Scripts/python -m spacy init config configs/gpu.cfg --lang en --pipeline ner --optimize accuracy --gpu
```
Expected: both files written. `gpu.cfg` uses `roberta-base` like the original models and needs `pip install spacy-transformers` plus a CUDA GPU to train. If generating `gpu.cfg` errors about `spacy-transformers`, run `.venv/Scripts/python -m pip install spacy-transformers` and retry. It's only needed for the GPU path and stays out of `requirements.txt`.

- [ ] **Step 6: Write `resume_ranker/cli.py` and `resume_ranker/__main__.py`**

`resume_ranker/__main__.py`:
```python
from .cli import main

main()
```

`resume_ranker/cli.py`:
```python
"""python -m resume_ranker {rank,train} ..."""
import argparse
import json
import sys
from pathlib import Path

from . import parse

ROOT = Path(__file__).resolve().parent.parent


def cmd_train(args):
    from spacy.cli.train import train

    data, out = ROOT / "data" / args.kind, parse.MODELS_DIR / args.kind
    train(
        ROOT / "configs" / ("gpu.cfg" if args.gpu else "cpu.cfg"), out,
        use_gpu=0 if args.gpu else -1,
        overrides={"paths.train": str(data / "train.spacy"), "paths.dev": str(data / "test.spacy")},
    )
    perf = json.loads((out / "model-best" / "meta.json").read_text(encoding="utf-8"))["performance"]
    print(f"{args.kind}: overall F1 {perf['ents_f']:.3f}")
    for label, s in sorted(perf["ents_per_type"].items()):
        print(f"  {label:<22} P {s['p']:.3f}  R {s['r']:.3f}  F1 {s['f']:.3f}")


def main(argv: list[str] | None = None):
    for stream in (sys.stdout, sys.stderr):  # resumes contain non-cp1252 characters on Windows
        getattr(stream, "reconfigure", lambda **_: None)(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="resume_ranker")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train an NER model from data/")
    t.add_argument("kind", choices=["resume", "jd"])
    t.add_argument("--gpu", action="store_true", help="transformer config; needs spacy-transformers + CUDA")
    t.set_defaults(fn=cmd_train)
    args = p.parse_args(argv)
    args.fn(args)
```

- [ ] **Step 7: Train both models** (CPU; JD takes a few minutes, resume roughly 10–30)

Run: `.venv/Scripts/python -m resume_ranker train jd`
Run: `.venv/Scripts/python -m resume_ranker train resume`
Expected: each ends with `overall F1 ...` plus a per-label table; `models/jd/model-best` and `models/resume/model-best` exist. CPU F1 will be below the old transformer models (0.58 JD / 0.86 resume). **Save both printed tables for the README (Task 9).**

- [ ] **Step 8: Smoke-check loading**

Run: `.venv/Scripts/python -c "from resume_ranker import parse; nlp = parse.load_nlp('resume'); print(parse.parse_resume(nlp, 'x', 'Top Skills\nPython\nDocker\nSoftware Engineer\nJanuary 2020 - Present (3 years 2 months)'))"`
Expected: a `Profile` with non-empty `skills`, printed without errors.

- [ ] **Step 9: Commit**

```bash
git add configs resume_ranker tests
git commit -m "Add CPU/GPU training command and model loading"
```

---

### Task 4: Hybrid scoring

**Files:**
- Create: `resume_ranker/score.py`, `tests/fakes.py`, `tests/test_score.py`

**Interfaces:**
- Consumes: `Profile` (Task 2).
- Produces:
  - `WEIGHTS: dict[str, float]` (keys in order `skills, experience, role, education, overall`), `SKILL_MATCH = 0.75`, `LEVEL_NAMES: dict[int, str]`
  - `Embed = Callable[[list[str]], np.ndarray]` returning L2-normalized rows; `load_embedder() -> Embed`
  - `Components` dataclass: `scores: dict[str, float]` (keys = `WEIGHTS` keys, same order), `matched: list[tuple[str, str]]` (jd_skill, resume_skill), `missing: list[str]`, `extra: list[str]`, `evidence: dict[str, str]` (same keys as scores)
  - `skill_match(jd_skills, cv_skills, embed) -> tuple[matched, missing, extra]`
  - `components(jd: Profile, cv: Profile, embed) -> Components`
  - `total(scores: dict[str, float], weights: dict[str, float]) -> float` (0–100; 0 if weights sum to 0)
  - `legacy_score(jd: Profile, cv: Profile) -> int` (notebook points, from `.ents`)

- [ ] **Step 1: Write the test fake** `tests/fakes.py`

```python
import numpy as np


class FakeEmbed:
    """Identical strings -> cosine 1, different -> 0; strings in the same synonym group -> 1."""

    def __init__(self, *synonyms):
        self.ids = {s: i for i, group in enumerate(synonyms) for s in group}
        self.next = len(synonyms)

    def __call__(self, texts):
        vecs = np.zeros((len(texts), 512))
        for row, t in enumerate(texts):
            if t not in self.ids:
                self.ids[t], self.next = self.next, self.next + 1
            vecs[row, self.ids[t]] = 1.0
        return vecs
```

- [ ] **Step 2: Write failing tests** `tests/test_score.py`

```python
import pytest
from fakes import FakeEmbed

from resume_ranker.parse import Profile
from resume_ranker.score import WEIGHTS, components, legacy_score, skill_match, total

JD = Profile("jd", "jd text", skills=["python", "docker", "react"], titles=["Backend Engineer"], years=3.0, degree=2)


def test_skill_match_uses_semantic_similarity():
    matched, missing, extra = skill_match(
        ["python", "react", "docker"], ["react native", "python", "excel"], FakeEmbed(["react", "react native"]))
    assert matched == [("python", "python"), ("react", "react native")]
    assert missing == ["docker"]
    assert extra == ["excel"]


def test_strong_candidate_scores_full_marks():
    cv = Profile("cv", "jd text", skills=["python", "docker", "react"], titles=["Backend Engineer"], years=5.0, degree=3)
    c = components(JD, cv, FakeEmbed())
    assert c.scores == pytest.approx(dict.fromkeys(WEIGHTS, 1.0))
    assert list(c.scores) == list(WEIGHTS) == list(c.evidence)
    assert total(c.scores, WEIGHTS) == pytest.approx(100)


def test_partial_candidate_gets_proportional_scores_and_evidence():
    cv = Profile("cv", "other text", skills=["python"], titles=["Teacher"], years=1.5, degree=2)
    c = components(JD, cv, FakeEmbed())
    assert c.scores == pytest.approx({"skills": 1 / 3, "experience": 0.5, "role": 0.0, "education": 1.0, "overall": 0.0})
    assert c.evidence["skills"] == "1/3 required skills"
    assert c.evidence["experience"] == "needs 3.0y, has 1.5y"


def test_lower_degree_gets_partial_education_credit():
    c = components(Profile("jd", "t", degree=3), Profile("cv", "t", degree=2), FakeEmbed())
    assert c.scores["education"] == pytest.approx(2 / 3)
    assert c.evidence["education"] == "needs master, has bachelor"


def test_jd_without_requirements_gives_full_credit():
    c = components(Profile("jd", "t", skills=["python"]), Profile("cv", "t", skills=["python"]), FakeEmbed())
    assert c.scores["experience"] == c.scores["role"] == c.scores["education"] == 1.0


def test_jd_without_skills_falls_back_to_overall_fit():  # Review Focus 2
    c = components(Profile("jd", "same text"), Profile("cv", "same text", skills=["python"]), FakeEmbed())
    assert c.scores["skills"] == c.scores["overall"] == 1.0
    assert "no skills found in JD" in c.evidence["skills"]
    assert c.extra == ["python"]


def test_resume_with_no_entities_scores_zero_without_crashing():  # Review Focus 3
    c = components(JD, Profile("cv", "unrelated"), FakeEmbed())
    assert c.scores == dict.fromkeys(WEIGHTS, 0.0)
    assert c.missing == ["python", "docker", "react"]
    assert c.evidence["experience"] == "needs 3.0y, not found"


def test_total_is_weighted_average_on_0_100():
    scores = {"skills": 1.0, "experience": 0.0, "role": 0.0, "education": 0.0, "overall": 0.0}
    assert total(scores, WEIGHTS) == pytest.approx(40)
    assert total(scores, {**WEIGHTS, "skills": 0.8}) == pytest.approx(100 * 0.8 / 1.4)


def test_total_with_all_weights_zero_is_zero():  # Review Focus 1
    assert total(dict.fromkeys(WEIGHTS, 1.0), dict.fromkeys(WEIGHTS, 0.0)) == 0.0


def test_legacy_score_reproduces_notebook_points():
    jd = Profile("jd", "", ents={"SKILLS": ["Python", "SQL"], "DEGREE": ["B.Tech"], "JOBPOST": ["Data Analyst"]})
    cv = Profile("cv", "", ents={"SKILLS": ["python", "Excel"], "DEGREE": ["b.tech"], "WORKED AS": ["Data Analyst", "Intern"]})
    assert legacy_score(jd, cv) == 5 + 10 + 10
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_score.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'resume_ranker.score'`.

- [ ] **Step 4: Implement** `resume_ranker/score.py`

```python
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
```

- [ ] **Step 5: Run tests**

Run: `.venv/Scripts/python -m pytest tests/test_score.py -v`
Expected: all PASS.

- [ ] **Step 6: Smoke-check the real embedder**

Run: `.venv/Scripts/python -c "from resume_ranker.score import load_embedder, skill_match; print(skill_match(['react', 'machine learning'], ['reactjs developer', 'ml'], load_embedder()))"`
Expected: prints a tuple without errors (downloads ~90MB on first run). Look at which pairs cleared 0.75; this is a sanity look, not an assertion.

- [ ] **Step 7: Commit**

```bash
git add resume_ranker/score.py tests/fakes.py tests/test_score.py
git commit -m "Add explainable hybrid scoring with legacy baseline"
```

---

### Task 5: Ranking with weights and must-haves

**Files:**
- Create: `resume_ranker/rank.py`, `tests/test_rank.py`

**Interfaces:**
- Consumes: `Profile`, `canonical_skill` (Task 2); `WEIGHTS`, `Components`, `components`, `total`, `Embed` (Task 4).
- Produces:
  - `Result` dataclass: `name: str, score: float, components: Components, disqualified: list[str]`
  - `score_all(jd: Profile, cvs: list[Profile], embed) -> list[tuple[str, Components]]` (the expensive step; cache this)
  - `rank(scored, weights=WEIGHTS, must_have=()) -> list[Result]` (cheap; disqualified last, then score desc, stable)
  - `to_rows(results) -> list[dict]` with keys `rank, name, score, <WEIGHTS keys>, disqualified, matched, missing, extra`

- [ ] **Step 1: Write failing tests** `tests/test_rank.py`

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_rank.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'resume_ranker.rank'`.

- [ ] **Step 3: Implement** `resume_ranker/rank.py`

```python
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
```

- [ ] **Step 4: Run all tests**

Run: `.venv/Scripts/python -m pytest -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add resume_ranker/rank.py tests/test_rank.py
git commit -m "Add ranking with tunable weights and must-have skills"
```

---

### Task 6: `rank` CLI command and sample data

**Files:**
- Create: `samples/jd.txt`, `samples/resumes/asha_verma.txt`, `samples/resumes/rohan_mehta.txt`, `samples/resumes/lena_fischer.txt`, `samples/resumes/tom_baker.txt`
- Modify: `resume_ranker/cli.py`

**Interfaces:**
- Consumes: `parse.read_all`, `parse.load_nlp`, `parse.parse_jd`, `parse.parse_resume`; `rank.score_all`, `rank.rank`, `rank.to_rows`; `score.load_embedder`.
- Produces: `python -m resume_ranker rank JD RESUME_DIR [--top N] [--csv PATH] [--must-have SKILL ...]`

- [ ] **Step 1: Write the sample files** (fictional people. They're written in the LinkedIn-PDF style the resume model was trained on.)

`samples/jd.txt`:
```
Senior Python Backend Engineer

We are looking for a Senior Python Backend Engineer to build and scale the APIs behind our analytics platform.

Requirements:
- 3+ years of professional experience in backend development
- Strong Python skills with Django or FastAPI
- Experience designing REST APIs
- Solid knowledge of PostgreSQL and Redis
- Hands-on experience with Docker and Kubernetes
- Familiarity with AWS and CI/CD pipelines
- Bachelor's degree in Computer Science or a related field

Nice to have:
- Experience with Celery and message queues
- Exposure to machine learning pipelines
```

`samples/resumes/asha_verma.txt`:
```
Contact
asha.verma@example.com
www.linkedin.com/in/asha-verma-example (LinkedIn)
Top Skills
Python
Django
PostgreSQL
Asha Verma
Senior Backend Engineer
Pune, Maharashtra, India
Summary
Backend engineer building high-traffic REST APIs with Python, Django and FastAPI.
Experience
Finlytics
Senior Backend Engineer
January 2021 - Present (3 years 9 months)
Designed REST APIs in Django and FastAPI backed by PostgreSQL and Redis, deployed with Docker and Kubernetes on AWS. Set up CI/CD with GitHub Actions.
Cartwise
Software Engineer
July 2019 - December 2020 (1 year 6 months)
Built Python microservices and Celery task queues.
Education
University of Pune
Bachelor of Engineering - BE, Computer Engineering · (2015 - 2019)
```

`samples/resumes/rohan_mehta.txt`:
```
Contact
rohan.mehta@example.com
www.linkedin.com/in/rohan-mehta-example (LinkedIn)
Top Skills
Java
Spring Boot
Docker
Rohan Mehta
Java Developer
Bengaluru, Karnataka, India
Summary
Java developer working on Spring Boot microservices.
Experience
Paystream
Java Developer
March 2022 - Present (2 years 7 months)
Built Spring Boot services with MySQL, containerized with Docker, deployed via Jenkins.
Education
Visvesvaraya Technological University
Bachelor of Technology, Computer Science · (2017 - 2021)
```

`samples/resumes/lena_fischer.txt`:
```
Contact
lena.fischer@example.com
www.linkedin.com/in/lena-fischer-example (LinkedIn)
Top Skills
Python
Machine Learning
Pandas
Lena Fischer
Data Scientist
Berlin, Germany
Summary
Data scientist building machine learning models and data pipelines in Python.
Experience
Retailytics GmbH
Data Scientist
April 2020 - Present (4 years 6 months)
Built forecasting models with scikit-learn and PyTorch, data pipelines with Pandas and SQL, served models through Flask APIs.
Education
Technical University of Munich
Master's degree, Data Science · (2018 - 2020)
```

`samples/resumes/tom_baker.txt`:
```
Contact
tom.baker@example.com
www.linkedin.com/in/tom-baker-example (LinkedIn)
Top Skills
React
JavaScript
CSS
Tom Baker
Frontend Developer
Manchester, United Kingdom
Summary
Frontend developer focused on accessible React interfaces.
Experience
Pixel Foundry
Junior Frontend Developer
September 2023 - Present (1 year 1 month)
Built React components with TypeScript, HTML and CSS.
```

- [ ] **Step 2: Add `cmd_rank` to `resume_ranker/cli.py`** and register it in `main()`

Add these imports at the top: `import csv`. Add the function:
```python
def cmd_rank(args):
    from .rank import rank, score_all, to_rows
    from .score import load_embedder

    jd_docs, warnings = parse.read_all([(args.jd.name, args.jd.read_bytes())])
    if not jd_docs:
        sys.exit(f"error: {warnings[0]}")
    cvs, more = parse.read_all((p.name, p.read_bytes()) for p in sorted(args.resumes.iterdir()) if p.is_file())
    for w in warnings + more:
        print(f"warning: {w}", file=sys.stderr)
    if not cvs:
        sys.exit("error: no readable resumes")

    jd = parse.parse_jd(parse.load_nlp("jd"), *jd_docs[0])
    if not jd.skills:
        print("warning: no skills found in JD; skills score uses overall fit", file=sys.stderr)
    resume_nlp = parse.load_nlp("resume")
    scored = score_all(jd, [parse.parse_resume(resume_nlp, *d) for d in cvs], load_embedder())
    rows = to_rows(rank(scored, must_have=args.must_have))

    for r in rows[: args.top]:
        flag = f"  DISQUALIFIED (missing {r['disqualified']})" if r["disqualified"] else ""
        print(f"{r['rank']:>3}. {r['score']:5.1f}  {r['name']}{flag}")
        print(f"       missing: {r['missing'] or '-'}")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {args.csv}")
```

In `main()`, before the `train` parser:
```python
    r = sub.add_parser("rank", help="rank resumes against a job description")
    r.add_argument("jd", type=Path, help="job description (.pdf or .txt)")
    r.add_argument("resumes", type=Path, help="folder of resumes (.pdf or .txt)")
    r.add_argument("--top", type=int, default=10)
    r.add_argument("--csv", type=Path, help="write the full ranking here")
    r.add_argument("--must-have", nargs="*", default=[], metavar="SKILL")
    r.set_defaults(fn=cmd_rank)
```

- [ ] **Step 3: Run end-to-end**

Run: `.venv/Scripts/python -m resume_ranker rank samples/jd.txt samples/resumes --csv out.csv`
Expected: 4 ranked lines and `wrote 4 rows to out.csv`. `asha_verma.txt` should rank first and `tom_baker.txt` last. If not, print the profiles (`parse.parse_resume(...)`) to see what NER extracted before touching weights.

- [ ] **Step 4: Check must-haves, bad files and the Windows console**

```bash
.venv/Scripts/python -m resume_ranker rank samples/jd.txt samples/resumes --must-have Kubernetes
printf 'not a pdf' > samples/resumes/broken.pdf
PYTHONIOENCODING=cp1252 .venv/Scripts/python -m resume_ranker rank samples/jd.txt samples/resumes --top 2
rm samples/resumes/broken.pdf
```
Expected: in the first run, every candidate without kubernetes is marked `DISQUALIFIED` and listed after the rest. In the second, a `warning: skipped broken.pdf: ...` line appears, the ranking still prints, and the `·` characters cause no `UnicodeEncodeError`.

- [ ] **Step 5: Run all tests and commit**

Run: `.venv/Scripts/python -m pytest`
Expected: all PASS.
```bash
git add samples resume_ranker/cli.py
git commit -m "Add rank CLI command and fictional sample data"
```

---

### Task 7: Streamlit app

**Files:**
- Create: `app.py`

**Interfaces:**
- Consumes: `parse.read_all`, `parse.load_nlp`, `parse.parse_jd`, `parse.parse_resume`; `rank.score_all`, `rank.rank`, `rank.to_rows`; `score.WEIGHTS`, `score.load_embedder`.

- [ ] **Step 1: Write `app.py`**

```python
"""Streamlit UI: upload a JD and resumes, tune weights, inspect and export the ranking."""
import pandas as pd
import streamlit as st

from resume_ranker import parse
from resume_ranker.rank import rank, score_all, to_rows
from resume_ranker.score import WEIGHTS, load_embedder

st.set_page_config(page_title="Resume Ranker", layout="wide")


@st.cache_resource(show_spinner="Loading models…")
def models():
    return parse.load_nlp("resume"), parse.load_nlp("jd"), load_embedder()


@st.cache_data(show_spinner="Scoring resumes…")
def score_uploads(jd_file: tuple[str, bytes], cv_files: tuple[tuple[str, bytes], ...]):
    """Expensive part, cached on file contents; slider changes only re-run rank()."""
    resume_nlp, jd_nlp, embed = models()
    jd_docs, warnings = parse.read_all([jd_file])
    cvs, more = parse.read_all(cv_files)
    if not jd_docs:
        return None, [], warnings + more
    jd = parse.parse_jd(jd_nlp, *jd_docs[0])
    return jd, score_all(jd, [parse.parse_resume(resume_nlp, *d) for d in cvs], embed), warnings + more


st.title("Resume Ranker")
st.caption("Ranks resumes against a job description using custom spaCy NER and semantic matching. Every score is explained.")
left, right = st.columns(2)
jd_up = left.file_uploader("Job description", type=["pdf", "txt"])
cv_up = right.file_uploader("Resumes", type=["pdf", "txt"], accept_multiple_files=True)
if not jd_up or not cv_up:
    st.info("Upload one job description and at least one resume. Try the files in `samples/`.")
    st.stop()

jd, scored, warnings = score_uploads((jd_up.name, jd_up.getvalue()), tuple((f.name, f.getvalue()) for f in cv_up))
for w in warnings:
    st.warning(w)
if jd is None or not scored:
    st.error("Nothing to rank.")
    st.stop()
if not jd.skills:
    st.warning("No skills found in the job description; the skills score falls back to overall fit.")

with st.sidebar:
    st.header("Weights")
    weights = {k: st.slider(k.capitalize(), 0.0, 1.0, v, 0.05) for k, v in WEIGHTS.items()}
    must_have = st.multiselect("Must-have skills", jd.skills)
    st.caption(f"JD asks for {jd.years:g}+ years" if jd.years else "JD states no years of experience")

results = rank(scored, weights, must_have)
df = pd.DataFrame(to_rows(results))
event = st.dataframe(
    df[["rank", "name", "score", *WEIGHTS, "disqualified"]],
    hide_index=True, on_select="rerun", selection_mode="single-row",
    column_config={
        "score": st.column_config.ProgressColumn("score", min_value=0, max_value=100, format="%.1f"),
        **{k: st.column_config.ProgressColumn(k, min_value=0, max_value=1, format="%.2f") for k in WEIGHTS},
    },
)
st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "ranking.csv", "text/csv")

picked = event.selection.rows
r = results[picked[0] if picked else 0]
st.subheader(f"{r.name}: {r.score:.1f}")
if r.disqualified:
    st.error("Missing must-have: " + ", ".join(r.disqualified))
matched = [f"- {j}" + (f" (as {c})" if c != j else "") for j, c in r.components.matched]
for col, title, lines in zip(
    st.columns(3), ["Matched", "Missing", "Extra"],
    [matched, [f"- {s}" for s in r.components.missing], [f"- {s}" for s in r.components.extra]],
):
    col.markdown(f"**{title}**\n\n" + ("\n".join(lines) or "-"))
st.table(pd.DataFrame({"score": r.components.scores, "evidence": r.components.evidence}))
```

- [ ] **Step 2: Run and check by hand**

Run: `.venv/Scripts/streamlit run app.py`
Expected, in the browser:
1. Upload `samples/jd.txt` and all 4 sample resumes; the table shows 4 rows with progress bars.
2. Drag the Experience slider to 1 and the others to 0; the order changes immediately with no "Scoring resumes…" spinner, which shows the cache works.
3. Set all sliders to 0; scores show 0.0 and nothing errors (Review Focus 1).
4. Pick `kubernetes` as a must-have; candidates without it move to the bottom with a "Missing must-have" banner when selected.
5. Click a row; the matched/missing/extra panel and the evidence table update.
6. Download CSV; it opens with all columns.
7. Also upload a `.txt` that is empty; a warning names it and the ranking still shows.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "Add Streamlit app with live re-weighting, gap report and CSV export"
```

---

### Task 8: Evaluation harness

**Files:**
- Create: `eval/evaluate.py`, `tests/test_evaluate.py`, `eval/labels.json` (generated by `pool`, then filled in by the human)

**Interfaces:**
- Consumes: `parse.load_nlp`, `parse.parse_jd`, `parse.parse_resume`; `rank.score_all`, `rank.rank`; `score.WEIGHTS`, `score.legacy_score`, `score.load_embedder`.
- Produces: `ndcg_at_k(rels: list[int], k=5) -> float`, `precision_at_k(rels, k=5, relevant=2) -> float`; CLI `python eval/evaluate.py {list,pool,score}`; `eval/labels.json` shaped `{"jd:<i>": {"resume:<j>": 0-3}}`.

- [ ] **Step 1: Write failing tests** `tests/test_evaluate.py`

```python
import pytest

from evaluate import ndcg_at_k, precision_at_k


def test_ndcg_perfect_order_is_one():
    assert ndcg_at_k([3, 2, 1, 0, 0]) == pytest.approx(1.0)


def test_ndcg_penalizes_bad_order():
    assert ndcg_at_k([0, 0, 1, 2, 3]) < ndcg_at_k([3, 0, 0, 2, 1]) < 1.0


def test_ndcg_only_looks_at_top_k():
    assert ndcg_at_k([3, 2, 1, 1, 1, 0], k=5) == pytest.approx(1.0)


def test_ndcg_all_irrelevant_is_zero():
    assert ndcg_at_k([0, 0, 0]) == 0.0


def test_precision_counts_relevance_two_and_up():
    assert precision_at_k([3, 1, 2, 0, 2, 3]) == pytest.approx(3 / 5)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluate'`.

- [ ] **Step 3: Implement** `eval/evaluate.py`

```python
"""Pooled evaluation: the original notebook score vs the hybrid (+ ablations) on hand-labeled held-out data.

  python eval/evaluate.py list           # show held-out JDs with their indices
  python eval/evaluate.py pool 0 5 12    # build labeling pools -> eval/pool/*.txt + eval/labels.json
  python eval/evaluate.py score          # after labeling every pair 0-3: NDCG@5 / P@5 table

ponytail: the held-out .spacy files double as the dev set used to pick model-best, so NER quality
on them is slightly optimistic. Split off a separate test set if that ever matters.
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
LABELS, POOL = ROOT / "eval" / "labels.json", ROOT / "eval" / "pool"
POOL_SIZE, TOP_FROM_EACH = 15, 7


def dcg(rels: list[int]) -> float:
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(rels: list[int], k: int = 5) -> float:
    ideal = dcg(sorted(rels, reverse=True)[:k])
    return dcg(rels[:k]) / ideal if ideal else 0.0


def precision_at_k(rels: list[int], k: int = 5, relevant: int = 2) -> float:
    return sum(r >= relevant for r in rels[:k]) / k


def held_out(kind: str) -> list[str]:
    import spacy
    from spacy.tokens import DocBin

    nlp = spacy.blank("en")
    return [d.text for d in DocBin().from_disk(ROOT / "data" / kind / "test.spacy").get_docs(nlp.vocab)]


def _parsed(jd_ids, cv_ids=None):
    from resume_ranker import parse
    from resume_ranker.score import load_embedder

    jd_texts, cv_texts = held_out("jd"), held_out("resume")
    jd_nlp, cv_nlp = parse.load_nlp("jd"), parse.load_nlp("resume")
    jds = {f"jd:{i}": parse.parse_jd(jd_nlp, f"jd:{i}", jd_texts[i]) for i in jd_ids}
    cvs = {f"resume:{i}": parse.parse_resume(cv_nlp, f"resume:{i}", cv_texts[i])
           for i in (range(len(cv_texts)) if cv_ids is None else cv_ids)}
    return jds, cvs, load_embedder()


def cmd_list(args):
    for i, text in enumerate(held_out("jd")):
        print(f"{i:>3}  {' '.join(text.split())[:100]}")


def cmd_pool(args):
    from resume_ranker.rank import rank, score_all
    from resume_ranker.score import legacy_score

    jds, cvs, embed = _parsed(args.jd)
    labels = json.loads(LABELS.read_text(encoding="utf-8")) if LABELS.exists() else {}
    POOL.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    for jd_name, jd in jds.items():
        legacy = sorted(cvs, key=lambda n: -legacy_score(jd, cvs[n]))[:TOP_FROM_EACH]
        hybrid = [r.name for r in rank(score_all(jd, list(cvs.values()), embed))][:TOP_FROM_EACH]
        pool = list(dict.fromkeys(legacy + hybrid))
        rest = [n for n in cvs if n not in pool]
        rng.shuffle(rest)
        pool += rest[: POOL_SIZE - len(pool)]
        (POOL / f"{jd_name.replace(':', '_')}.txt").write_text(jd.text, encoding="utf-8")
        entry = labels.setdefault(jd_name, {})
        for n in pool:
            entry.setdefault(n, None)
            (POOL / f"{n.replace(':', '_')}.txt").write_text(cvs[n].text, encoding="utf-8")
    LABELS.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    print(f"Texts in {POOL}. Replace every null in {LABELS} with 0-3 "
          "(3 = would interview, 2 = plausible, 1 = weak, 0 = irrelevant), then run: score")


def cmd_score(args):
    from resume_ranker.rank import rank, score_all
    from resume_ranker.score import WEIGHTS, legacy_score

    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    unlabeled = sum(v is None for pool in labels.values() for v in pool.values())
    if unlabeled:
        sys.exit(f"{unlabeled} pairs in {LABELS} are still null; label them 0-3 first.")
    jd_ids = [int(k.split(":")[1]) for k in labels]
    cv_ids = sorted({int(n.split(":")[1]) for pool in labels.values() for n in pool})
    jds, cvs, embed = _parsed(jd_ids, cv_ids)
    systems = {"legacy (notebook)": None, "hybrid": WEIGHTS,
               **{f"hybrid without {k}": {**WEIGHTS, k: 0.0} for k in WEIGHTS}}
    sums = {name: [0.0, 0.0] for name in systems}
    for jd_name, rels in labels.items():
        jd, pool = jds[jd_name], [cvs[n] for n in rels]
        scored = score_all(jd, pool, embed)
        for name, weights in systems.items():
            if weights is None:
                order = [cv.name for cv in sorted(pool, key=lambda cv: -legacy_score(jd, cv))]
            else:
                order = [r.name for r in rank(scored, weights)]
            ranked = [rels[n] for n in order]
            sums[name][0] += ndcg_at_k(ranked)
            sums[name][1] += precision_at_k(ranked)
    print(f"{len(labels)} JDs, {sum(map(len, labels.values()))} labeled pairs\n")
    print("| system | NDCG@5 | P@5 |\n|---|---|---|")
    for name, (n, p) in sums.items():
        print(f"| {name} | {n / len(labels):.3f} | {p / len(labels):.3f} |")


def main():
    p = argparse.ArgumentParser(prog="evaluate")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    pool = sub.add_parser("pool")
    pool.add_argument("jd", type=int, nargs="+", help="held-out JD indices from `list`")
    pool.set_defaults(fn=cmd_pool)
    sub.add_parser("score").set_defaults(fn=cmd_score)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python -m pytest -v`
Expected: all PASS.

- [ ] **Step 5: Build pools**

Run: `PYTHONIOENCODING=utf-8 .venv/Scripts/python eval/evaluate.py list`
Pick 3 JDs with distinct roles (e.g. one ML, one web/backend, one other) and pass their indices:
Run: `.venv/Scripts/python eval/evaluate.py pool <i> <j> <k>`
Expected: `eval/pool/` holds 3 JD texts and up to 45 resume texts; `eval/labels.json` has 3 × 15 `null` entries.

- [ ] **Step 6: HUMAN GATE: label the pools**

The human partner reads `eval/pool/jd_<i>.txt` and each resume text, then replaces every `null` in `eval/labels.json` with 0–3. **The implementer must not fill in labels.** Fake labels would make the whole result meaningless. Stop here and ask.

- [ ] **Step 7: Score and commit**

Run: `.venv/Scripts/python eval/evaluate.py score`
Expected: a markdown table with rows for legacy, hybrid and the 5 ablations. Save it for the README. If the hybrid doesn't beat legacy, report that honestly; look at the ablation rows before changing any weights.
```bash
git add eval/evaluate.py eval/labels.json tests/test_evaluate.py
git commit -m "Add pooled NDCG/P@5 evaluation against the notebook baseline"
```

---

### Task 9: README, model hosting, deployment

**Files:**
- Modify: `README.md` (full rewrite)

**Interfaces:**
- Consumes: the training F1 tables (Task 3 Step 7) and the eval table (Task 8 Step 7).

- [ ] **Step 1: Rewrite `README.md`**

Write the content below. Replace the two marked tables with the real output saved in Task 3 Step 7 and Task 8 Step 7.

````markdown
# Resume Ranker: custom spaCy NER + semantic matching

Rank a folder of resumes against a job description and see **why** each candidate ranked where they did.

![app screenshot](docs/screenshot.png)

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt   # .venv/bin on Linux/macOS
python -m resume_ranker train jd && python -m resume_ranker train resume          # CPU, ~15-40 min total
python -m resume_ranker rank samples/jd.txt samples/resumes --csv ranking.csv
streamlit run app.py
```
Set `RESUME_RANKER_HF_REPO=<user>/<repo>` to download pretrained models instead of training.

## How it works

1. **Extract**: two custom spaCy NER models (trained here on ~700 labeled resumes and ~160 JDs) pull out skills, job titles, experience, and degrees. A curated skill dictionary backs up the weaker JD model.
2. **Normalize**: skill aliases (`ReactJS` → `react`), per-job durations summed into total years, degrees mapped to levels.
3. **Score**: five explainable components, each in [0, 1]:

| Component | Default weight | How |
|---|---|---|
| Skills | 0.40 | Each JD skill matched to the closest resume skill by MiniLM embedding (cosine ≥ 0.75) |
| Experience | 0.20 | Total years vs the JD requirement, capped at 1 |
| Role | 0.15 | Best embedding match between past titles and the JD title |
| Education | 0.10 | Degree level vs the lowest acceptable level |
| Overall fit | 0.15 | Whole-document embedding similarity |

4. **Rank**: weighted total on 0–100. Must-have skills disqualify a candidate without hiding them. In the app the weights are sliders and re-ranking is instant.

## Results

Pooled evaluation on held-out data: 3 JDs × 15 resumes, hand-labeled 0–3 (`python eval/evaluate.py score`):

<!-- paste the table printed by Task 8 Step 7 here -->

NER quality (CPU tok2vec models; the original roberta models scored F1 0.863 on resumes and 0.583 on JDs):

<!-- paste the per-label F1 tables printed by Task 3 Step 7 here -->

## Limitations

- Training data is mostly LinkedIn-export PDFs of tech profiles; other layouts and domains extract worse.
- Overlapping jobs double-count experience; scanned PDFs (no text layer) are skipped, not OCR'd.
- 45 labeled pairs is enough to compare systems, not to claim production accuracy.
- The original Colab notebook is kept in `notebooks/` for reference.
````

When the README is final, the two `<!-- paste ... -->` comment lines must be gone, replaced by the real tables.

- [ ] **Step 2: Take the screenshot**

Run the app with the samples, select the top candidate, and save a screenshot to `docs/screenshot.png`.

- [ ] **Step 3: HUMAN GATE: publish models to Hugging Face Hub** (outward-facing; the human runs it or explicitly approves)

```bash
.venv/Scripts/huggingface-cli login
.venv/Scripts/python -c "from huggingface_hub import HfApi; import os; api = HfApi(); repo = os.environ['HF_REPO']; api.create_repo(repo, exist_ok=True); api.upload_folder(repo_id=repo, folder_path='models', allow_patterns=['*/model-best/**'])"
```
with `HF_REPO=<their-username>/resume-ranker-models` set. Verify: delete `models/`, set `RESUME_RANKER_HF_REPO`, and re-run the Task 6 Step 3 command. Expected: models download and the ranking matches.

- [ ] **Step 4: HUMAN GATE: deploy the app** (outward-facing)

Push the branch to GitHub, then on share.streamlit.io create an app from the repo with main file `app.py` and secret `RESUME_RANKER_HF_REPO = "<their-username>/resume-ranker-models"` (top-level secrets are exposed as env vars). Add the live URL to the top of the README.

- [ ] **Step 5: Final verification and commit**

Run: `.venv/Scripts/python -m pytest` → all PASS.
Run: `git grep -n "paste the" -- README.md` → no output.
```bash
git add README.md docs/screenshot.png
git commit -m "Rewrite README with results, quickstart and limitations"
```
