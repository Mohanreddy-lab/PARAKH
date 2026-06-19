"""
jd_parser.py — Stage 1: JD Intelligence

Reads a job description and extracts structured data using the local LLM.
Returns a ParsedJD object with explicit_skills, implied_skills, seniority,
latent_needs, and a plain-English summary.

Works offline with Ollama — no API key needed.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(__file__))
from llm import get_llm

load_dotenv()

# ── Data model ────────────────────────────────────────────────────────────────

class Skill(BaseModel):
    name:       str
    importance: str = "required"   # "required" | "preferred" | "nice-to-have"

class ParsedJD(BaseModel):
    role_title:      str       = "Unknown Role"
    seniority:       str       = "mid"
    explicit_skills: List[Skill] = Field(default_factory=list)
    implied_skills:  List[Skill] = Field(default_factory=list)
    latent_needs:    List[str]   = Field(default_factory=list)
    summary:         str       = ""

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert technical recruiter analyzing a job description.

Return ONLY a JSON object with exactly these keys:
  "role_title"      : job title as stated in the JD
  "seniority"       : "junior", "mid", "senior", or "lead"
  "required_skills" : list of skills explicitly required (lowercase, plain names)
  "implied_skills"  : skills clearly needed but NOT stated in JD (lowercase)
  "latent_needs"    : list of 2-3 short phrases describing what this role really tests
  "summary"         : one sentence for a recruiter describing what the role needs

Start your reply with {{ and end with }}. No markdown, no explanation outside JSON."""

USER_TEMPLATE = """Job Description:
{jd_text}

JSON:"""

# ── Parsing helpers ───────────────────────────────────────────────────────────

def _safe_parse(raw: str) -> dict:
    """Strip markdown fences, try json.loads, fall back to regex extract."""
    # Remove ```json ... ``` fences
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r'```\s*$', '', cleaned.strip())

    # Direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Extract first {...}
    m = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # Give up — return safe defaults
    print("[jd_parser] WARNING: could not parse LLM output; using defaults.")
    return {}


def _to_parsed_jd(data: dict) -> ParsedJD:
    """Convert raw JSON dict to ParsedJD with Skill objects."""
    def _skills(raw_list, importance) -> List[Skill]:
        if not isinstance(raw_list, list):
            return []
        return [
            Skill(name=str(s).lower().strip(), importance=importance)
            for s in raw_list if s
        ]

    return ParsedJD(
        role_title=str(data.get("role_title", "Unknown Role")).strip(),
        seniority=str(data.get("seniority", "mid")).lower().strip(),
        explicit_skills=_skills(data.get("required_skills", []), "required"),
        implied_skills=_skills(data.get("implied_skills",  []), "preferred"),
        latent_needs=[str(n) for n in data.get("latent_needs", []) if n],
        summary=str(data.get("summary", "")).strip(),
    )

# ── Core function ─────────────────────────────────────────────────────────────

def parse_jd(jd_text: str) -> ParsedJD:
    """Send the JD text to the local LLM and return a ParsedJD object."""
    llm    = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",  USER_TEMPLATE),
    ])
    chain    = prompt | llm
    response = chain.invoke({"jd_text": jd_text})
    data     = _safe_parse(response.content)
    return _to_parsed_jd(data)

# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_jd(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"JD file is empty: {path}")
    return text

def save_parsed_jd(parsed: ParsedJD, out_path: str | Path) -> None:
    Path(out_path).write_text(
        json.dumps(parsed.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[jd_parser] Saved -> {out_path}")

def load_parsed_jd(path: str | Path) -> ParsedJD:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ParsedJD(**data)

# ── CLI / smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    SAMPLE_JD = """Senior Data Engineer -- FinTech Platform

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
- Experience with Airflow or Prefect for orchestration

Responsibilities include mentoring junior engineers and owning the data model design."""

    print("Parsing JD with local LLM...\n")
    parsed = parse_jd(SAMPLE_JD)

    print(f"Role      : {parsed.role_title}")
    print(f"Seniority : {parsed.seniority}")
    print(f"Summary   : {parsed.summary}")
    print("\nRequired skills:")
    for s in parsed.explicit_skills:
        print(f"  - {s.name}")
    print("\nImplied skills:")
    for s in parsed.implied_skills:
        print(f"  - {s.name}")
    print("\nLatent needs:")
    for n in parsed.latent_needs:
        print(f"  - {n}")
