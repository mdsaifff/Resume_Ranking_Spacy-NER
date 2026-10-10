import pymupdf
import pytest
import spacy

from resume_ranker import parse
from resume_ranker.parse import (
    canonical_skill, degree_levels, dictionary_skills, duration_years, entities, jd_profile, parse_jd,
    parse_resume, read_all, read_text, required_years, resume_profile,
)


def test_read_text_txt_handles_utf8_and_bom():
    assert read_text("﻿héllo · world".encode("utf-8"), "a.txt") == "héllo · world"


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
    ("Minimum 4 years� experience", 4), ("7+ years of experience with PHP", 7),
    ("3 to 5 years", 3), ("strong communication", None),
])
def test_required_years(s, need):
    assert required_years(s) == need


@pytest.mark.parametrize("s,levels", [
    ("Bachelor's degree, Computer Science", [2]),
    ("Doctor of Philosophy (Ph.D.), Watershed Management", [4]),
    ("M.S. or Ph.D. in engineering", [4, 3]),
    ("Bachelor�s/Master�s degree in Engineering", [3, 2]),
    ("M.Tech/B.tech/B.E/BCA in Computer Science", [3, 2]),
    ("High School Diploma, Physical Sciences", [1]),
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


def test_parse_resume_adds_skills_mentioned_in_job_descriptions():
    nlp = spacy.blank("en")
    nlp.add_pipe("entity_ruler").add_patterns([{"label": "SKILLS", "pattern": "Python"}])
    cv = parse_resume(nlp, "cv.txt", "Top Skills\nPython\nDeployed services with Docker and Kubernetes on AWS.")
    assert cv.skills == ["python", "docker", "kubernetes", "aws"]
