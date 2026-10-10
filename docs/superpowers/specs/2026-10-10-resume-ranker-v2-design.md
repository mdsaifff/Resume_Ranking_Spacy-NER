# Resume Ranker → portfolio-grade project (design)

## Context
`Resume_Ranking_Spacy-NER` is a Colab notebook export (`resume_ranker(spacy).py`) with two custom spaCy NER models (resume F1 0.86, JD F1 0.58). Today it:
- only runs in Colab (`!pip`, Drive mount, `/content/drive/...` paths);
- **cannot load its models**: repo has only `config.cfg`/`meta.json`/`tokenizer`, no `ner/`/`transformer/` weights (train/test `.spacy` data *is* committed);
- scores **one** resume vs one JD with exact-string matching (+5/skill, +10 degree/role/company); experience ignored; company list is hardcoded and buggy (`'tcS'`); no ranking.

Goal (user-confirmed): a **portfolio project** focused on resume ranking that a reviewer can run in ~2 min. It should produce an explainable ranked shortlist and show measured improvement over the old scoring.

Decisions made: retrain models · Streamlit app + CLI · hybrid NER + semantic scoring · add-ons: evaluation harness, tunable weights + must-haves, skill-gap report + CSV export. Out of scope: PII/bias masking, FastAPI, React, LLM scoring, auth, DB.

## Architecture (flat package, few files)
```
resume_ranker/
  parse.py     # file -> text (PyMuPDF for PDF, plain .txt); NER -> {label: [str]}; skill PhraseMatcher fallback; years regex
  score.py     # component scores, weighted total, explanation dict; legacy_score() = old notebook logic (eval baseline)
  rank.py      # rank(jd, resumes, weights, must_have) -> sorted list[Result]
  cli.py       # `python -m resume_ranker rank JD RESUME_DIR [--top N] [--csv out.csv]` and `train {resume,jd} [--gpu]`
  skills.json  # canonical skill -> aliases ("react": ["reactjs","react.js"], ...)
app.py         # Streamlit UI
configs/       # resume_cpu.cfg, jd_cpu.cfg (tok2vec, CPU) + existing transformer cfgs (GPU)
eval/          # labels.json (JD↔resume relevance 0-3) + evaluate.py
tests/         # test_parse.py, test_score.py (pytest, no fixtures framework)
requirements.txt, README.md, .gitignore (models/)
```
The notebook is moved to `notebooks/` as history. The `Model/` training `.spacy` files stay, moved to `data/`.

## Scoring (the core)
For each resume vs JD, each component returns a score in [0,1] plus its evidence:
| Component | How | Evidence shown |
|---|---|---|
| Skills | Normalize via `skills.json`; per JD skill take max cosine vs resume skills (sentence-transformers `all-MiniLM-L6-v2`, CPU); matched if ≥ 0.75; score = matched/required | matched / missing / extra |
| Experience | Regex years from JD `EXPERIENCE` and resume `YEARS OF EXPERIENCE` ("3+ years", "2-4 yrs"); `min(1, have/need)`; 1 if JD states none | "needs 3y, has 5y" |
| Role | max cosine of resume `WORKED AS`/`DESIGNATION` vs JD `JOBPOST` | best-matching title |
| Education | degree level ordinal (diploma<bachelor<master<phd) keyword map; 1 if ≥ required | levels |
| Overall fit | cosine of whole-document embeddings | — |

`total = 100 * Σ wᵢ·sᵢ / Σ wᵢ`. Default weights: skills .4, experience .2, role .15, education .1, overall .15. **Must-have skills**: a missing one marks the candidate `disqualified` and sorts them to the bottom, with the reason shown. They stay visible.
JD skills = NER `SKILLS` ∪ PhraseMatcher hits from `skills.json` (covers the weak JD model). The company-prestige list is dropped as an arbitrary, biased signal.
`# ponytail:` thresholds/weights are constants; tune via eval harness, not config files.

## Streamlit app
Upload 1 JD (PDF/txt) + N resumes → sidebar weight sliders + must-have multiselect (populated from extracted JD skills) → ranked table (score, component bars). Clicking a row opens the skill-gap panel (matched/missing/extra plus evidence). Download CSV. Models are cached with `st.cache_resource`. Re-weighting reuses the cached component scores, so only the weighted sum is recomputed.

## Models & training
`cli train resume|jd` wraps `spacy.cli.train` with the committed `.spacy` data. CPU tok2vec configs by default (any laptop), and `--gpu` uses the existing roberta transformer configs. Trained models go to `models/` (gitignored) and get published to Hugging Face Hub. The app/CLI downloads them via `huggingface_hub.snapshot_download` if `models/` is missing. README reports per-label F1 from `meta.json` for both variants.

## Evaluation harness
`eval/labels.json`: ~3 JDs × ~15 resumes, relevance 0–3, **hand-labeled by the user** (resume texts come from the held-out `test_data.spacy` docs). `eval/evaluate.py` prints NDCG@5 and P@5 for `legacy_score` vs the hybrid, plus an ablation (hybrid minus each component). The results table goes in the README.

## Error handling
Unreadable or empty file → skip it with a warning in the UI/CLI and keep ranking the rest. JD with zero extracted skills → warning, and the skills weight falls back to the overall-fit score. Scanned PDFs (no text layer): warn only (OCR is out of scope).

## Testing
`tests/test_parse.py`: years regex cases, skill alias normalization, PhraseMatcher fallback. `tests/test_score.py`: component math, weight normalization, must-have disqualification, ranking order on tiny hand-built entity dicts (no model load; embedder stubbed with a fake). Run: `pytest`.

## Verification (end-to-end)
1. `pip install -r requirements.txt && python -m resume_ranker train resume && python -m resume_ranker train jd` (CPU) → `models/` populated, F1 printed.
2. `pytest` passes.
3. `python -m resume_ranker rank samples/jd.txt samples/resumes --top 5 --csv out.csv` → ranked table + CSV.
4. `streamlit run app.py` → upload, move sliders (order changes), set must-have (disqualifies), download CSV.
5. `python eval/evaluate.py` → hybrid NDCG@5 > legacy (after user labels data).

## Suggested build order
1. Package skeleton and file→text parsing; CPU training command and models.
2. Scoring and ranking with tests; CLI.
3. Streamlit app (weights, must-haves, gap report, CSV).
4. Evaluation harness (needs the user's labels); README with results; HF Hub upload; deploy to HF Spaces.
