"""Shared fixtures for PARAKH tests."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jd_parser import ParsedJD, Skill


@pytest.fixture
def sample_jd() -> ParsedJD:
    return ParsedJD(
        role_title="Senior Data Engineer",
        seniority="senior",
        explicit_skills=[
            Skill(name="python",        importance="required"),
            Skill(name="apache spark",  importance="required"),
            Skill(name="aws",           importance="required"),
            Skill(name="dbt",           importance="preferred"),
            Skill(name="apache kafka",  importance="nice-to-have"),
        ],
        implied_skills=[
            Skill(name="sql",           importance="required"),
            Skill(name="git",           importance="preferred"),
            Skill(name="linux",         importance="nice-to-have"),
        ],
        summary="Senior data engineer to build and own data pipelines on AWS.",
    )


@pytest.fixture
def sample_profiles():
    return [
        {
            "candidate_id": "C001", "title": "Senior Data Engineer",
            "skills": ["Python", "PySpark", "Airflow", "AWS", "dbt", "SQL"],
            "summary": "8 years on AWS Spark pipelines.",
            "_embed_score": 0.91, "_recall_rank": 1,
        },
        {
            "candidate_id": "C002", "title": "Junior Frontend Developer",
            "skills": ["React", "JavaScript", "CSS"],
            "summary": "Frontend developer with React.",
            "_embed_score": 0.31, "_recall_rank": 2,
        },
        {
            "candidate_id": "C003", "title": "Data Engineer",
            "skills": ["Spark", "Python", "AWS"],
            "summary": "Self-taught data engineer. Ships fast.",
            "_embed_score": 0.78, "_recall_rank": 3,
            "github_stars": 150,
        },
        {
            "candidate_id": "C004", "title": "Lead Data Platform Engineer",
            "skills": ["Python", "Spark", "Kafka", "dbt", "AWS", "Kubernetes"],
            "summary": "Lead engineer. Built lakehouse. Mentors team.",
            "_embed_score": 0.72, "_recall_rank": 4,
            "endorsements": 80,
        },
    ]
