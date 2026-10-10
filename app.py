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
