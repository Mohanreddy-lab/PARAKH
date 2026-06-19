"""
api.py — PARAKH REST API

FastAPI server exposing the full 5-stage pipeline over HTTP.

Endpoints:
  POST /api/v1/rank       run the pipeline, returns ranked candidates
  GET  /health            liveness check
  GET  /api/v1/models     list models available in the local Ollama instance
  GET  /api/v1/synonyms   list all skill synonym groups

Run:
  uvicorn src.api:app --reload        (from project root)
  uvicorn api:app    --reload --port 8000   (from src/)
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agent  import run_pipeline, DEMO_JD, DEMO_PROFILES
from skills import SKILL_SYNONYMS

app = FastAPI(
    title="PARAKH Candidate Ranker",
    description="Ranks candidates against a job description using 5-stage AI pipeline.",
    version="1.0.0",
)


# ── Request / Response models ─────────────────────────────────────────────────

class RankRequest(BaseModel):
    jd_text:  str            = Field(..., description="Job description plain text")
    profiles: Optional[list] = Field(None, description="List of candidate profile dicts. Uses demo data if omitted.")
    rerank_n: int            = Field(8,   ge=1, le=200, description="Candidates to LLM-rerank")
    force:    bool           = Field(False, description="Bypass stage caches")


class CandidateResult(BaseModel):
    rank:            int
    candidate_id:    str
    score_100:       float
    final_score:     float
    llm_score:       int
    confidence:      str
    hidden_gem:      bool
    reason:          str
    skill_score:     float
    seniority_score: float
    activity_score:  float
    composite_score: float
    embedding_score: float


class RankResponse(BaseModel):
    status:      str
    elapsed_s:   float
    total:       int
    hidden_gems: int
    candidates:  List[dict]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    model = os.getenv("PARAKH_MODEL", "llama3.2")
    provider = os.getenv("LLM_PROVIDER", "ollama")
    return {"status": "ok", "llm_provider": provider, "model": model}


@app.post("/api/v1/rank", response_model=RankResponse)
def rank(req: RankRequest):
    """Run the full PARAKH pipeline and return ranked candidates."""
    data_dir = Path(__file__).parent.parent / "data"
    profiles = req.profiles or DEMO_PROFILES
    jd_text  = req.jd_text or DEMO_JD

    t0 = time.time()
    try:
        ranked = run_pipeline(
            jd_text=jd_text,
            profiles=profiles,
            rerank_n=req.rerank_n,
            out_dir=data_dir,
            force=req.force,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed   = round(time.time() - t0, 2)
    gems      = sum(1 for c in ranked if c.get("hidden_gem"))

    # Sanitise for JSON (remove internal _ fields)
    clean = []
    for rank_i, c in enumerate(ranked, 1):
        entry = {k: v for k, v in c.items() if not k.startswith("_")}
        entry["rank"]         = rank_i
        entry["score_100"]    = c.get("score_100", round(c.get("final_score", 0) * 100, 1))
        entry["hidden_gem"]   = bool(c.get("hidden_gem", False))
        entry["skill_evidence"] = c.get("skill_evidence", {})
        clean.append(entry)

    return RankResponse(
        status="ok",
        elapsed_s=elapsed,
        total=len(ranked),
        hidden_gems=gems,
        candidates=clean,
    )


@app.get("/api/v1/models")
def list_models():
    """List models available in the local Ollama instance."""
    try:
        import ollama
        models = [m["model"] for m in ollama.list()["models"]]
        current = os.getenv("PARAKH_MODEL", "llama3.2")
        return {"current": current, "available": models}
    except Exception as exc:
        return {"current": os.getenv("PARAKH_MODEL", "llama3.2"),
                "available": [], "error": str(exc)}


@app.get("/api/v1/synonyms")
def list_synonyms(skill: Optional[str] = None):
    """List skill synonym groups (optionally filter by skill name)."""
    if skill:
        key = skill.lower().strip()
        matches = {k: v for k, v in SKILL_SYNONYMS.items()
                   if key in k or any(key in s for s in v)}
        return {"query": skill, "matches": matches}
    return {"total": len(SKILL_SYNONYMS), "synonyms": SKILL_SYNONYMS}


# ── Dev entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
