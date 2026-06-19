"""
rerank.py — Stage 4: Honest LLM Rerank

Scores top candidates with GPT-4o using structured output (Pydantic).
Chain-of-thought reasoning is in the system prompt; structured output
ensures clean JSON every time.

Confidence weighting: low-confidence scores drift toward the composite
(safe default) rather than overriding it.

  Final = BLEND_COMPOSITE * composite + BLEND_LLM * effective_llm
  effective_llm = conf_w * llm_norm + (1 - conf_w) * composite

Provides both a blocking function and a streaming generator so the
Streamlit demo can show results live as each candidate is scored.
"""

import os
import sys
import time
from typing import List, Dict, Generator

from tqdm import tqdm
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    CONFIDENCE_WEIGHT, BLEND_COMPOSITE, BLEND_LLM,
    LLM_RETRIES, OPENAI_MODEL, MAX_PROFILE_CHARS, RERANK_N,
)


class RankResult(BaseModel):
    llm_score:  int = Field(ge=1, le=10, description="Candidate fit score 1-10")
    reason:     str = Field(description="One sentence with evidence from the profile. Start with 'Limited evidence:' if thin.")
    confidence: str = Field(description="'high', 'medium', or 'low'")


SYSTEM_PROMPT = """You are an honest technical recruiter scoring candidate fit for a job.

Rules:
- llm_score: integer 1-10 based ONLY on evidence present in the profile text.
- reason: one sentence using REAL words from the profile. If evidence is thin, start with "Limited evidence:"
- confidence: "high" (clear direct evidence), "medium" (some signals, some gaps), "low" (thin profile).

NEVER invent skills or experiences not stated in the profile."""

HUMAN_PROMPT = """Job summary:
{jd_summary}

Candidate profile:
{profile_text}

Score this candidate."""


def _make_jd_summary(parsed_jd) -> str:
    explicit = [s.name for s in parsed_jd.explicit_skills]
    implied  = [s.name for s in parsed_jd.implied_skills]
    return (
        f"Role: {parsed_jd.role_title}. "
        f"Seniority: {parsed_jd.seniority}. "
        f"Must-have: {', '.join(explicit[:8])}. "
        f"Implied: {', '.join(implied[:5])}. "
        f"{parsed_jd.summary}"
    )


def _make_profile_text(profile: dict, max_chars: int) -> str:
    priority = [
        "title", "current_role", "headline",
        "summary", "bio", "about",
        "skills", "tech_skills", "tools",
        "experience", "work_history",
        "education", "certifications",
    ]
    lines, total = [], 0
    for field in priority:
        val = profile.get(field)
        if not val:
            continue
        if isinstance(val, list):
            val = ", ".join(str(v) for v in val)
        line = f"{field}: {str(val).strip()}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)


def _final_score(composite: float, llm_score: int, confidence: str) -> float:
    llm_norm  = llm_score / 10.0
    conf_w    = CONFIDENCE_WEIGHT.get(confidence, 0.5)
    effective = conf_w * llm_norm + (1.0 - conf_w) * composite
    return round(BLEND_COMPOSITE * composite + BLEND_LLM * effective, 4)


def _get_chain():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set. Set it in .env or environment.")
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, api_key=api_key)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",  HUMAN_PROMPT),
    ])
    return prompt | llm.with_structured_output(RankResult)


def _score_one(candidate: dict, chain, jd_summary: str) -> dict:
    profile_text = _make_profile_text(candidate, MAX_PROFILE_CHARS)
    last_exc = None

    for attempt in range(LLM_RETRIES + 1):
        try:
            result: RankResult = chain.invoke({
                "jd_summary":   jd_summary,
                "profile_text": profile_text,
            })
            llm_data = {
                "llm_score":  result.llm_score,
                "reason":     result.reason,
                "confidence": result.confidence.lower(),
            }
            break
        except Exception as exc:
            last_exc = exc
            if attempt < LLM_RETRIES:
                time.sleep(2 ** attempt)
    else:
        cid = candidate.get("candidate_id", candidate.get("id", "?"))
        print(f"\n[rerank] All retries failed for {cid}: {last_exc}")
        llm_data = {
            "llm_score":  0,
            "reason":     "Model call failed — scored 0 to avoid false ranking.",
            "confidence": "low",
        }

    entry = dict(candidate)
    entry.update(llm_data)
    entry["final_score"] = _final_score(
        entry.get("composite_score", 0.0),
        entry["llm_score"],
        entry["confidence"],
    )
    return entry


def rerank_candidates(
    candidates: List[Dict],
    parsed_jd,
    top_n: int = None,
) -> List[Dict]:
    """Score top_n candidates and return sorted by final_score. Blocking."""
    results = list(rerank_stream(candidates, parsed_jd, top_n))
    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results


def rerank_stream(
    candidates: List[Dict],
    parsed_jd,
    top_n: int = None,
) -> Generator[Dict, None, None]:
    """Generator: yield each candidate result as soon as GPT-4o finishes it."""
    if top_n is None:
        top_n = RERANK_N

    batch      = candidates[:top_n]
    chain      = _get_chain()
    jd_summary = _make_jd_summary(parsed_jd)

    print(f"[rerank] Scoring {len(batch)} candidates with {OPENAI_MODEL}")
    print(f"         Confidence weights: high=1.0, medium=0.70, low=0.30\n")

    for candidate in tqdm(batch, desc="Reranking", unit="candidate"):
        yield _score_one(candidate, chain, jd_summary)


if __name__ == "__main__":
    from pathlib import Path
    data_dir   = Path(__file__).parent.parent / "data"
    jd_path    = data_dir / "parsed_jd.json"
    top50_path = data_dir / "top50.json"

    from jd_parser import load_parsed_jd
    from scoring   import load_scored

    parsed_jd  = load_parsed_jd(jd_path)
    candidates = load_scored(top50_path)

    print(f"Reranking top 5 with {OPENAI_MODEL}...\n")
    results = rerank_candidates(candidates, parsed_jd, top_n=5)

    for i, c in enumerate(results, 1):
        cid = c.get("candidate_id", c.get("id", "?"))
        gem = " * GEM" if c.get("hidden_gem") else ""
        print(f"{i}. [{cid}]{gem}")
        print(f"   composite={c['composite_score']}  llm={c['llm_score']}/10  "
              f"conf={c['confidence']}  final={c['final_score']}")
        print(f"   {c['reason']}")
        print()
