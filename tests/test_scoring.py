"""Tests for scoring.py — multi-signal composite scoring."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from scoring import (
    score_candidates, _infer_seniority, _seniority_score,
    _activity_score_raw, _normalise,
)


class TestNormalise:
    def test_basic(self):
        result = _normalise([0.0, 0.5, 1.0])
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(1.0)

    def test_all_identical_returns_half(self):
        result = _normalise([0.7, 0.7, 0.7])
        assert all(v == 0.5 for v in result)


class TestInferSeniority:
    def test_lead(self):
        assert _infer_seniority({"title": "Lead Data Engineer"}) == "lead"

    def test_senior(self):
        assert _infer_seniority({"title": "Senior ML Engineer"}) == "senior"

    def test_junior(self):
        assert _infer_seniority({"title": "Junior Developer"}) == "junior"

    def test_default_mid(self):
        assert _infer_seniority({"title": "Data Engineer"}) == "mid"


class TestSeniorityScore:
    def test_exact_match(self):
        assert _seniority_score("senior", "senior") == 1.0

    def test_above_required(self):
        assert _seniority_score("lead", "senior") == 1.0

    def test_one_below(self):
        assert _seniority_score("mid", "senior") == pytest.approx(0.55)

    def test_two_below(self):
        assert _seniority_score("junior", "senior") == pytest.approx(0.15)


class TestActivityScore:
    def test_no_activity_returns_zero(self):
        assert _activity_score_raw({}) == 0.0

    def test_endorsements(self):
        score = _activity_score_raw({"endorsements": 50, "github_stars": 100})
        assert score > 0

    def test_skips_non_numeric(self):
        score = _activity_score_raw({"endorsements": "n/a", "github_stars": 80})
        assert score == pytest.approx(80.0)


class TestScoreCandidates:
    def test_returns_sorted_by_composite(self, sample_jd, sample_profiles):
        scored = score_candidates(sample_profiles, sample_jd)
        scores = [c["composite_score"] for c in scored]
        assert scores == sorted(scores, reverse=True)

    def test_fields_present(self, sample_jd, sample_profiles):
        scored = score_candidates(sample_profiles, sample_jd)
        for c in scored:
            assert "embedding_score"  in c
            assert "skill_score"      in c
            assert "seniority_score"  in c
            assert "activity_score"   in c
            assert "composite_score"  in c
            assert "hidden_gem"       in c
            assert "skill_evidence"   in c

    def test_scores_in_range(self, sample_jd, sample_profiles):
        scored = score_candidates(sample_profiles, sample_jd)
        for c in scored:
            assert 0.0 <= c["composite_score"] <= 1.0

    def test_senior_engineer_ranks_above_frontend(self, sample_jd, sample_profiles):
        scored = score_candidates(sample_profiles, sample_jd)
        ids = [c["candidate_id"] for c in scored]
        assert ids.index("C001") < ids.index("C002")

    def test_hidden_gem_flagged(self, sample_jd, sample_profiles):
        scored = score_candidates(sample_profiles, sample_jd)
        # At least one candidate should have hidden_gem as bool
        assert all(isinstance(c["hidden_gem"], bool) for c in scored)

    def test_skill_evidence_has_required_fields(self, sample_jd, sample_profiles):
        scored = score_candidates(sample_profiles, sample_jd)
        for c in scored:
            ev = c["skill_evidence"]
            assert "required_matched" in ev
            assert "required_missing" in ev

    def test_lead_engineer_seniority_advantage(self, sample_jd, sample_profiles):
        scored = score_candidates(sample_profiles, sample_jd)
        lead   = next(c for c in scored if c["candidate_id"] == "C004")
        junior = next(c for c in scored if c["candidate_id"] == "C002")
        assert lead["seniority_score"] >= junior["seniority_score"]
