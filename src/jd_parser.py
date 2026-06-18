# Stage 1 — JD Intelligence
# Reads a job description and extracts: required skills, implied skills,
# and seniority level. Returns structured data used by all later stages.

import json
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

# ── Data model ────────────────────────────────────────────────────────────────

class Skill(BaseModel):
    name: str = Field(description="Skill name, lowercase and concise")
    importance: str = Field(description="'required', 'preferred', or 'nice-to-have'")

class ParsedJD(BaseModel):
    role_title: str = Field(description="Job title as stated in the JD")
    seniority: str = Field(description="'junior', 'mid', 'senior', or 'lead'")
    explicit_skills: List[Skill] = Field(
        description="Skills the JD directly mentions"
    )
    implied_skills: List[Skill] = Field(
        description="Skills the role clearly needs but the JD did not spell out"
    )
    summary: str = Field(
        description="Two-sentence plain-English summary of what this role really needs"
    )

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert technical recruiter.
Given a job description, extract structured data about the role.

Rules:
- explicit_skills: only skills directly named in the JD text.
- implied_skills: skills any qualified professional would need for this role
  even if not mentioned (e.g., a "Senior Backend Engineer" role implies
  system design, code review, mentoring).
- seniority: infer from titles, years of experience asked, and responsibilities.
- importance: mark a skill 'required' if the JD uses words like "must", "required",
  "essential"; 'preferred' if it says "nice to have", "bonus", "plus"; otherwise
  use your judgement.
- summary: write for a recruiter who has 10 seconds to understand the role.
- Be concise. Do not invent facts not present or inferable from the JD."""

HUMAN_PROMPT = """Job Description:
{jd_text}

Extract the structured data now."""

# ── Core function ─────────────────────────────────────────────────────────────

def parse_jd(jd_text: str) -> ParsedJD:
    """Send the JD text to GPT-4o and return a ParsedJD object."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set. Export it before running.")

    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)
    structured_llm = llm.with_structured_output(ParsedJD)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])

    chain = prompt | structured_llm
    result = chain.invoke({"jd_text": jd_text})
    return result

# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_jd(path: str | Path) -> str:
    """Read a job description from a plain-text or .txt file."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"JD file is empty: {path}")
    return text

def save_parsed_jd(parsed: ParsedJD, out_path: str | Path) -> None:
    """Save the parsed JD as a JSON file for use by later stages."""
    Path(out_path).write_text(
        json.dumps(parsed.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved parsed JD → {out_path}")

def load_parsed_jd(path: str | Path) -> ParsedJD:
    """Load a previously saved ParsedJD JSON back into a ParsedJD object."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ParsedJD(**data)

# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_dir = Path(__file__).parent.parent / "data"
    jd_file = data_dir / "job_description.txt"
    out_file = data_dir / "parsed_jd.json"

    if not jd_file.exists():
        # Write a sample JD so the stage can be tested immediately
        sample = """Senior Data Engineer — FinTech Platform

We are looking for a Senior Data Engineer to join our growing data platform team.

Requirements:
- 5+ years of experience in data engineering
- Strong proficiency in Python and SQL
- Experience with Apache Spark and distributed data processing
- Hands-on with cloud platforms (AWS preferred: S3, Glue, Redshift)
- Familiarity with dbt or similar transformation tools
- Must have experience designing and maintaining data pipelines

Nice to have:
- Kafka or other streaming technologies
- Exposure to ML pipelines and feature stores
- Experience with Airflow or Prefect for orchestration

Responsibilities include mentoring junior engineers, owning the data model design,
and collaborating with product and ML teams to deliver reliable data products."""
        jd_file.write_text(sample, encoding="utf-8")
        print(f"No JD found — wrote sample to {jd_file}")

    if len(sys.argv) > 1:
        jd_file = Path(sys.argv[1])

    print(f"Parsing: {jd_file}")
    jd_text = load_jd(jd_file)
    parsed = parse_jd(jd_text)
    save_parsed_jd(parsed, out_file)

    # Pretty-print for quick review
    print("\n── Parsed JD ──────────────────────────────")
    print(f"Role   : {parsed.role_title}")
    print(f"Level  : {parsed.seniority}")
    print(f"Summary: {parsed.summary}")
    print("\nExplicit skills:")
    for s in parsed.explicit_skills:
        print(f"  [{s.importance:12s}] {s.name}")
    print("\nImplied skills:")
    for s in parsed.implied_skills:
        print(f"  [{s.importance:12s}] {s.name}")
