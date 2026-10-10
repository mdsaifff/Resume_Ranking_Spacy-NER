# Resume Ranker: custom spaCy NER + semantic matching

Rank a folder of resumes against a job description and see **why** each candidate ranked where they did.

![app screenshot](docs/screenshot.png)

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt   # .venv/bin on Linux/macOS
python -m resume_ranker train jd && python -m resume_ranker train resume          # CPU
python -m resume_ranker rank samples/jd.txt samples/resumes --csv ranking.csv
streamlit run app.py
```
Set `RESUME_RANKER_HF_REPO=<user>/<repo>` to download pretrained models instead of training.

```
$ python -m resume_ranker rank samples/jd.txt samples/resumes --must-have kubernetes
  1.  86.3  asha_verma.txt
       missing: backend, machine learning
  2.  49.5  lena_fischer.txt  DISQUALIFIED (missing kubernetes)
  ...
```

## How it works

1. **Extract**: two custom spaCy NER models (trained here on ~700 labeled resumes and ~160 JDs) pull out skills, job titles, experience and degrees. A curated skill dictionary (`resume_ranker/skills.json`) also scans both documents. It catches skills the models miss, e.g. tools named only inside a job description.
2. **Normalize**: skill aliases (`ReactJS` → `react`), per-job durations ("3 years 9 months") summed into total years, JD requirements ("3-5 years", "5<10 years") read as a lower bound, degrees mapped to levels.
3. **Score**: five explainable components, each in [0, 1]:

| Component | Default weight | How |
|---|---|---|
| Skills | 0.40 | Each JD skill matched to the closest resume skill by MiniLM embedding (cosine ≥ 0.75) |
| Experience | 0.20 | Total years vs the JD requirement, capped at 1 |
| Role | 0.15 | Best embedding match between past titles and the JD title |
| Education | 0.10 | Degree level vs the lowest acceptable level |
| Overall fit | 0.15 | Whole-document embedding similarity |

4. **Rank**: weighted total on 0–100. Must-have skills disqualify a candidate without hiding them. In the app the weights are sliders, and re-ranking is instant because component scores are cached.

The old notebook's scoring (exact string matches, +5/+10 points, a hardcoded "good companies" list) is kept as `legacy_score` and serves as the evaluation baseline. The company list was dropped from scoring as an arbitrary and biased signal.

## Results

Pooled evaluation on held-out data: 3 JDs × 15 resumes, hand-labeled 0–3 (`python eval/evaluate.py score`):

<!-- paste the table printed by Task 8 Step 7 here -->

NER quality (CPU tok2vec models; the original roberta models scored F1 0.863 on resumes and 0.583 on JDs):

<!-- paste the per-label F1 tables printed by Task 3 Step 7 here -->

## Project layout

| Path | What |
|---|---|
| `resume_ranker/parse.py` | file → text, NER, normalization into `Profile`s |
| `resume_ranker/score.py` | component scores, evidence, weighted total, legacy baseline |
| `resume_ranker/rank.py` | ranking under weights and must-haves, CSV rows |
| `resume_ranker/cli.py` | `rank` and `train` commands |
| `app.py` | Streamlit UI |
| `eval/evaluate.py` | pooled NDCG@5 / P@5 evaluation |
| `data/` | spaCy training data (resume and JD NER) |
| `configs/` | `cpu.cfg` (default) and `gpu.cfg` (roberta-base; needs `spacy-transformers` + CUDA in a separate env, it conflicts with `sentence-transformers`) |
| `notebooks/` | the original Colab notebook, for reference |

Run the tests with `python -m pytest`.

## Limitations

- Training data is mostly LinkedIn-export PDFs of tech profiles; other layouts and domains extract worse.
- Overlapping jobs double-count experience; scanned PDFs (no text layer) are skipped, not OCR'd.
- MiniLM reads only the first ~256 tokens of each document for the overall-fit score.
- 45 labeled pairs is enough to compare systems, not to claim production accuracy.
