"""Integration tests — full pipeline with mocked LLM (no API key)."""

import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import patch, MagicMock
from pathlib import Path
import pytest

import json
from jd_parser import ParsedJD, Skill
from scoring   import score_candidates
from rerank    import rerank_candidates
from output    import write_output, normalize_scores


def _mock_response(score=8, reason="Strong match.", confidence="high"):
    r = MagicMock()
    r.content = json.dumps({"llm_score": score, "reason": reason, "confidence": confidence})
    return r


@pytest.fixture
def full_jd():
    return ParsedJD(
        role_title="Senior Data Engineer",
        seniority="senior",
        explicit_skills=[
            Skill(name="python",        importance="required"),
            Skill(name="apache spark",  importance="required"),
            Skill(name="aws",           importance="required"),
            Skill(name="dbt",           importance="preferred"),
        ],
        implied_skills=[
            Skill(name="sql",   importance="required"),
            Skill(name="linux", importance="nice-to-have"),
        ],
        summary="Senior data engineer to build AWS pipelines.",
    )


@pytest.fixture
def full_profiles():
    return [
        {
            "candidate_id": "C001", "title": "Senior Data Engineer",
            "skills": ["Python", "PySpark", "AWS", "dbt", "SQL"],
            "_embed_score": 0.91, "_recall_rank": 1,
        },
        {
            "candidate_id": "C002", "title": "Junior Frontend Developer",
            "skills": ["React", "JavaScript"],
            "_embed_score": 0.30, "_recall_rank": 4,
        },
        {
            "candidate_id": "C003", "title": "Data Engineer",
            "skills": ["Spark", "Python", "AWS"],
            "_embed_score": 0.80, "_recall_rank": 2,
            "github_stars": 200,
        },
        {
            "candidate_id": "C004", "title": "Lead Data Platform Engineer",
            "skills": ["Python", "Spark", "Kafka", "dbt", "AWS", "Kubernetes", "Airflow"],
            "_embed_score": 0.72, "_recall_rank": 3,
            "endorsements": 90,
        },
    ]


class TestScoringToOutput:
    def test_full_score_and_write(self, full_jd, full_profiles):
        scored = score_candidates(full_profiles, full_jd)
        normalize_scores(scored)
        with tempfile.TemporaryDirectory() as tmp:
            csv_p, json_p = write_output(scored, out_dir=tmp)
            assert csv_p.exists()
            data = json.loads(json_p.read_text())
            assert len(data) == 4
            assert data[0]["rank"] == 1

    def test_c001_ranks_above_c002(self, full_jd, full_profiles):
        scored = score_candidates(full_profiles, full_jd)
        ids = [c["candidate_id"] for c in scored]
        assert ids.index("C001") < ids.index("C002")

    def test_hidden_gem_bool(self, full_jd, full_profiles):
        scored = score_candidates(full_profiles, full_jd)
        assert all(isinstance(c["hidden_gem"], bool) for c in scored)


class TestScoringToRerankPipeline:
    def test_rerank_preserves_skill_evidence(self, full_jd, full_profiles):
        scored = score_candidates(full_profiles, full_jd)

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = _mock_response()
        with patch("rerank._get_chain", return_value=mock_chain):
            ranked = rerank_candidates(scored, full_jd, top_n=4)

        for c in ranked:
            assert "skill_evidence"   in c
            assert "final_score"      in c
            assert "composite_score"  in c

    def test_final_scores_sorted(self, full_jd, full_profiles):
        scored = score_candidates(full_profiles, full_jd)

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = [
            _mock_response(9,  "Top.",   "high"),
            _mock_response(6,  "Mid.",   "medium"),
            _mock_response(3,  "Weak.",  "low"),
            _mock_response(10, "Great.", "high"),
        ]
        with patch("rerank._get_chain", return_value=mock_chain):
            ranked = rerank_candidates(scored, full_jd, top_n=4)

        scores = [c["final_score"] for c in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_end_to_end_csv_output(self, full_jd, full_profiles):
        import csv
        scored = score_candidates(full_profiles, full_jd)

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = _mock_response(7, "Test.", "medium")
        with patch("rerank._get_chain", return_value=mock_chain):
            ranked = rerank_candidates(scored, full_jd, top_n=4)

        normalize_scores(ranked)
        with tempfile.TemporaryDirectory() as tmp:
            csv_p, _ = write_output(ranked, out_dir=tmp)
            with csv_p.open() as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 4
            assert rows[0]["rank"] == "1"
            assert float(rows[0]["final_score"]) > 0
