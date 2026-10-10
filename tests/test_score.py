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
