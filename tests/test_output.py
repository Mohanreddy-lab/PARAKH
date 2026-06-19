"""Tests for output.py — CSV/JSON writing and leaderboard."""

import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathlib import Path
import pytest
from output import write_output, normalize_scores, _get_id, _conf_color


SAMPLE = [
    {
        "candidate_id": "C001", "title": "Senior Data Engineer",
        "final_score": 0.85, "llm_score": 9, "confidence": "high",
        "hidden_gem": False, "reason": "Strong AWS Spark pipeline experience.",
        "skill_score": 0.80, "seniority_score": 1.0, "activity_score": 0.5,
        "composite_score": 0.75, "embedding_score": 0.91,
    },
    {
        "candidate_id": "C004", "title": "Data Engineer",
        "final_score": 0.62, "llm_score": 6, "confidence": "medium",
        "hidden_gem": True, "reason": "Limited evidence: self-taught but ships Spark.",
        "skill_score": 0.50, "seniority_score": 0.55, "activity_score": 0.0,
        "composite_score": 0.55, "embedding_score": 0.78,
    },
]


class TestGetId:
    def test_candidate_id_preferred(self):
        assert _get_id({"candidate_id": "C001", "id": "X"}) == "C001"

    def test_fallback_to_id(self):
        assert _get_id({"id": "X"}) == "X"

    def test_unknown_when_empty(self):
        assert _get_id({}) == "UNKNOWN"


class TestConfColor:
    def test_high_green(self):
        assert _conf_color("high") == "green"

    def test_low_red(self):
        assert _conf_color("low") == "red"

    def test_unknown_white(self):
        assert _conf_color("unknown") == "white"


class TestNormalizeScores:
    def test_adds_score_100(self):
        data = [{"final_score": 0.85}]
        normalize_scores(data)
        assert data[0]["score_100"] == pytest.approx(85.0, abs=0.1)


class TestWriteOutput:
    def test_creates_csv_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_p, json_p = write_output(SAMPLE, out_dir=tmp)
            assert csv_p.exists()
            assert json_p.exists()

    def test_csv_has_all_columns(self):
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            csv_p, _ = write_output(SAMPLE, out_dir=tmp)
            with csv_p.open() as f:
                reader = csv.DictReader(f)
                row = next(reader)
                assert "rank" in row
                assert "candidate_id" in row
                assert "final_score" in row
                assert "hidden_gem" in row

    def test_ranks_start_at_1(self):
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            csv_p, _ = write_output(SAMPLE, out_dir=tmp)
            with csv_p.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert rows[0]["rank"] == "1"
                assert rows[1]["rank"] == "2"

    def test_hidden_gem_column(self):
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            csv_p, _ = write_output(SAMPLE, out_dir=tmp)
            with csv_p.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert rows[0]["hidden_gem"] == "no"
                assert rows[1]["hidden_gem"] == "yes"

    def test_json_has_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, json_p = write_output(SAMPLE, out_dir=tmp)
            data = json.loads(json_p.read_text())
            assert data[0]["rank"] == 1
            assert data[1]["rank"] == 2

    def test_json_strips_internal_fields(self):
        sample = [dict(SAMPLE[0])]
        sample[0]["_rank_jump"] = 3
        with tempfile.TemporaryDirectory() as tmp:
            _, json_p = write_output(sample, out_dir=tmp)
            data = json.loads(json_p.read_text())
            assert "_rank_jump" not in data[0]
