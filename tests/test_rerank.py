"""Tests for rerank.py — mocked LLM so no API key needed."""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import patch, MagicMock
import pytest
from rerank import (
    _final_score, _make_jd_summary, _make_profile_text, _parse_result,
    rerank_candidates, rerank_stream,
)


class TestFinalScore:
    def test_high_confidence_weights_llm(self):
        score = _final_score(0.5, 10, "high")
        assert score > 0.5 * 0.40 + 1.0 * 0.60 * 0.40   # LLM should dominate

    def test_low_confidence_drifts_to_composite(self):
        # Low confidence: LLM score very different from composite, should move less
        high_llm_high_conf = _final_score(0.5, 10, "high")
        high_llm_low_conf  = _final_score(0.5, 10, "low")
        assert high_llm_low_conf < high_llm_high_conf

    def test_range_zero_to_one(self):
        for score in [_final_score(0.5, s, c)
                      for s in [1, 5, 10]
                      for c in ["high", "medium", "low"]]:
            assert 0.0 <= score <= 1.0


class TestMakeJDSummary:
    def test_contains_role(self, sample_jd):
        summary = _make_jd_summary(sample_jd)
        assert "Senior Data Engineer" in summary

    def test_contains_skills(self, sample_jd):
        summary = _make_jd_summary(sample_jd)
        assert "python" in summary.lower()

    def test_contains_seniority(self, sample_jd):
        summary = _make_jd_summary(sample_jd)
        assert "senior" in summary.lower()


class TestMakeProfileText:
    def test_includes_title(self):
        profile = {"title": "Data Engineer", "skills": ["Python", "Spark"]}
        text = _make_profile_text(profile, max_chars=500)
        assert "Data Engineer" in text

    def test_respects_max_chars(self):
        profile = {
            "title": "Engineer",
            "skills": ["Python"] * 100,
            "summary": "x" * 500,
        }
        text = _make_profile_text(profile, max_chars=200)
        assert len(text) <= 300   # some slack for line formatting

    def test_list_skills_joined(self):
        profile = {"skills": ["Python", "Spark", "AWS"]}
        text = _make_profile_text(profile, max_chars=500)
        assert "Python" in text
        assert "Spark"  in text


class TestParseResult:
    def test_parses_valid_json(self):
        raw = json.dumps({"llm_score": 8, "reason": "Strong match.", "confidence": "high"})
        result = _parse_result(raw)
        assert result["llm_score"]  == 8
        assert result["confidence"] == "high"

    def test_clamps_score_to_range(self):
        raw = json.dumps({"llm_score": 15, "reason": "x", "confidence": "low"})
        assert _parse_result(raw)["llm_score"] == 10

    def test_falls_back_on_garbage(self):
        result = _parse_result("this is not json at all")
        assert 1 <= result["llm_score"] <= 10   # fallback default


def _make_mock_chain(score=8, reason="Strong evidence.", confidence="high"):
    mock_response = MagicMock()
    mock_response.content = json.dumps({"llm_score": score, "reason": reason, "confidence": confidence})
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_response
    return mock_chain


class TestReRankCandidatesMocked:
    def test_returns_sorted_by_final_score(self, sample_jd, sample_profiles):
        with patch("rerank._get_chain") as mock_get:
            mock_get.return_value = _make_mock_chain()
            # Add required composite_score to profiles
            for p in sample_profiles:
                p.setdefault("composite_score", 0.5)
            results = rerank_candidates(sample_profiles, sample_jd, top_n=4)
        scores = [c["final_score"] for c in results]
        assert scores == sorted(scores, reverse=True)

    def test_all_fields_present(self, sample_jd, sample_profiles):
        with patch("rerank._get_chain") as mock_get:
            mock_get.return_value = _make_mock_chain()
            for p in sample_profiles:
                p.setdefault("composite_score", 0.5)
            results = rerank_candidates(sample_profiles, sample_jd, top_n=2)
        for c in results:
            assert "llm_score"   in c
            assert "reason"      in c
            assert "confidence"  in c
            assert "final_score" in c

    def test_stream_yields_per_candidate(self, sample_jd, sample_profiles):
        with patch("rerank._get_chain") as mock_get:
            mock_get.return_value = _make_mock_chain()
            for p in sample_profiles:
                p.setdefault("composite_score", 0.5)
            results = list(rerank_stream(sample_profiles, sample_jd, top_n=3))
        assert len(results) == 3

    def test_retry_on_failure(self, sample_jd, sample_profiles):
        call_count = [0]
        ok_response = MagicMock()
        ok_response.content = json.dumps({"llm_score": 7, "reason": "OK", "confidence": "medium"})

        def flaky_invoke(inputs):
            call_count[0] += 1
            if call_count[0] < 2:
                raise RuntimeError("API timeout")
            return ok_response

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = flaky_invoke

        with patch("rerank._get_chain", return_value=mock_chain):
            with patch("rerank.time.sleep"):
                for p in sample_profiles[:1]:
                    p.setdefault("composite_score", 0.5)
                results = list(rerank_stream(sample_profiles[:1], sample_jd, top_n=1))

        assert results[0]["llm_score"] == 7
        assert call_count[0] >= 2
