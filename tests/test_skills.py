"""Tests for skills.py — synonym expansion and matching."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from skills import canonical, expand_skill, skill_matches, matched_skills


class TestCanonical:
    def test_spark_synonyms_map_to_canonical(self):
        assert canonical("pyspark") == "apache spark"
        assert canonical("spark sql") == "apache spark"

    def test_aws_synonyms(self):
        assert canonical("amazon web services") == "aws"

    def test_unknown_passes_through(self):
        assert canonical("zoobaz") == "zoobaz"


class TestExpandSkill:
    def test_expand_includes_synonyms(self):
        variants = expand_skill("apache spark")
        assert "pyspark"    in variants
        assert "spark sql"  in variants
        assert "spark"      in variants

    def test_expand_sql_includes_postgres(self):
        variants = expand_skill("sql")
        assert "postgresql" in variants
        assert "mysql"      in variants


class TestSkillMatches:
    def test_exact_match(self):
        assert skill_matches("python", "Experienced Python developer.")

    def test_synonym_match_pyspark(self):
        assert skill_matches("apache spark", "Uses PySpark on AWS daily.")

    def test_parenthetical_aws(self):
        assert skill_matches("aws (s3, glue)", "Worked on AWS infrastructure.")

    def test_no_match(self):
        assert not skill_matches("tensorflow", "Writes React components.")

    def test_word_boundary_no_false_positive(self):
        # "r" skill should not match "recruiter"
        assert not skill_matches("r", "Looking for a recruiter with management skills.")

    def test_kafka_alias(self):
        assert skill_matches("apache kafka", "Built streaming systems with Kafka.")


class TestMatchedSkills:
    def test_returns_matched_and_missing(self):
        text = "Python developer with PySpark and Airflow on AWS."
        matched, missing = matched_skills(
            ["python", "apache spark", "apache airflow", "dbt"],
            text,
        )
        assert "python"         in matched
        assert "apache spark"   in matched
        assert "apache airflow" in matched
        assert "dbt"            in missing
        assert len(matched) + len(missing) == 4
