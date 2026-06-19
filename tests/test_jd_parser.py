"""Tests for jd_parser.py — mocked LLM so no API key needed."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import patch, MagicMock
import json
import pytest
from jd_parser import ParsedJD, Skill, save_parsed_jd, load_parsed_jd
import tempfile
from pathlib import Path


SAMPLE_JD = """
Senior Data Engineer -- FinTech

Requirements:
- 5+ years Python and SQL
- Apache Spark (PySpark)
- AWS (S3, Glue, Redshift)
- dbt experience preferred

Nice to have: Kafka, Airflow
"""


def _make_parsed_jd():
    return ParsedJD(
        role_title="Senior Data Engineer",
        seniority="senior",
        explicit_skills=[
            Skill(name="python",       importance="required"),
            Skill(name="apache spark", importance="required"),
            Skill(name="aws",          importance="required"),
            Skill(name="dbt",          importance="preferred"),
        ],
        implied_skills=[
            Skill(name="sql",   importance="required"),
            Skill(name="linux", importance="nice-to-have"),
        ],
        summary="Senior data engineer to own AWS pipelines.",
    )


class TestParsedJDModel:
    def test_fields_exist(self):
        p = _make_parsed_jd()
        assert p.role_title == "Senior Data Engineer"
        assert p.seniority  == "senior"
        assert len(p.explicit_skills) == 4
        assert len(p.implied_skills)  == 2

    def test_skill_importance_values(self):
        p = _make_parsed_jd()
        importances = {s.importance for s in p.explicit_skills}
        assert importances <= {"required", "preferred", "nice-to-have"}

    def test_model_dump_roundtrip(self):
        p = _make_parsed_jd()
        d = p.model_dump()
        p2 = ParsedJD(**d)
        assert p2.role_title == p.role_title


class TestSaveLoadParsedJD:
    def test_roundtrip(self):
        p = _make_parsed_jd()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "parsed_jd.json"
            save_parsed_jd(p, path)
            loaded = load_parsed_jd(path)
            assert loaded.role_title == p.role_title
            assert loaded.seniority  == p.seniority
            assert len(loaded.explicit_skills) == len(p.explicit_skills)

    def test_skills_preserved(self):
        p = _make_parsed_jd()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "parsed_jd.json"
            save_parsed_jd(p, path)
            loaded = load_parsed_jd(path)
            names = [s.name for s in loaded.explicit_skills]
            assert "python" in names
            assert "aws"    in names


class TestParseJDMocked:
    def test_parse_jd_calls_llm(self):
        import json
        fake_json = json.dumps({
            "role_title": "Senior Data Engineer",
            "seniority": "senior",
            "required_skills": ["python", "apache spark", "aws"],
            "implied_skills": ["sql", "linux"],
            "latent_needs": ["owns pipelines end-to-end"],
            "summary": "Senior data engineer to build AWS pipelines.",
        })
        mock_response = MagicMock()
        mock_response.content = fake_json

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response

        with patch("jd_parser.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_get_llm.return_value = mock_llm

            with patch("jd_parser.ChatPromptTemplate") as mock_prompt:
                mock_pt = MagicMock()
                mock_prompt.from_messages.return_value = mock_pt
                mock_pt.__or__ = lambda s, o: mock_chain

                from jd_parser import parse_jd
                result = parse_jd(SAMPLE_JD)
                # Verify it parsed into a ParsedJD
                assert result.seniority == "senior"
