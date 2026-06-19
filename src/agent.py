"""
agent.py — Pipeline Orchestrator

Runs the full PARAKH 5-stage pipeline end-to-end:
  Stage 1: JD parsing      (GPT-4o structured output)
  Stage 2: Fast recall     (sentence-transformers + FAISS)
  Stage 3: Multi-signal scoring
  Stage 4: LLM rerank      (GPT-4o with confidence weighting)
  Stage 5: Output          (CSV + JSON + Rich terminal)

Cache: each stage's result is pickled under data/.cache/ using an
MD5 hash of its inputs as the key. Re-runs skip unchanged stages.
Use --force to bypass all caches.
"""

import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, os.path.dirname(__file__))

from config    import RECALL_K, SCORE_N, RERANK_N, EMBED_MODEL
from jd_parser import parse_jd, load_jd, save_parsed_jd, load_parsed_jd, ParsedJD
from recall    import load_profiles, profile_to_text, embed_texts, build_index, recall_top_k
from scoring   import score_candidates, save_scored
from rerank    import rerank_candidates
from output    import write_output, print_summary, normalize_scores


DEMO_PROFILES: List[Dict] = [
    {
        "candidate_id": "C001", "name": "Arjun Mehta",
        "title": "Senior Data Engineer",
        "skills": ["Python", "PySpark", "Airflow", "AWS", "S3", "dbt", "SQL", "Glue", "Redshift"],
        "summary": "8 years building data pipelines on AWS. Led Hadoop to S3/Glue migration. Mentors 4-person team.",
        "education": "B.Tech Computer Science, IIT Delhi",
        "experience": "Led team of 4 engineers. Designed star-schema warehouse on Redshift.",
        "endorsements": 72,
    },
    {
        "candidate_id": "C002", "name": "Priya Singh",
        "title": "Junior Frontend Developer",
        "skills": ["React", "JavaScript", "CSS", "HTML", "TypeScript"],
        "summary": "2 years building React UIs. Strong in frontend performance optimization.",
        "education": "B.Sc IT",
    },
    {
        "candidate_id": "C003", "name": "Rahul Verma",
        "title": "Data Analyst",
        "skills": ["SQL", "Python", "Tableau", "Excel"],
        "summary": "Strong SQL skills and data storytelling. No pipeline experience.",
        "education": "MBA Analytics",
    },
    {
        "candidate_id": "C004", "name": "Kavya Nair",
        "title": "Data Engineer",
        "skills": ["Spark", "Airflow", "Python", "AWS", "Kafka"],
        "summary": "Built batch and streaming pipelines on AWS. Self-taught. No CS degree. Ships fast.",
        "education": "B.E. Electronics",
        "github_stars": 120,
    },
    {
        "candidate_id": "C005", "name": "Siddharth Rao",
        "title": "ML Engineer",
        "skills": ["Python", "TensorFlow", "PyTorch", "SQL", "AWS", "Airflow", "Spark"],
        "summary": "ML engineer with strong Python and pipeline experience. Deployed models to prod on AWS.",
        "education": "M.Tech AI, IIT Bombay",
    },
    {
        "candidate_id": "C006", "name": "Deepa Krishnan",
        "title": "Senior Analytics Engineer",
        "skills": ["dbt", "SQL", "Python", "Snowflake", "Airflow", "data modeling"],
        "summary": "5 years transforming data with dbt and SQL. Expert in dimensional modeling on Snowflake.",
        "education": "B.Com + data engineering certification",
        "endorsements": 45,
    },
    {
        "candidate_id": "C007", "name": "Vikram Sharma",
        "title": "Analytics Engineer",
        "skills": ["dbt", "SQL", "Snowflake", "Python", "Spark", "AWS"],
        "summary": "Analytics engineer transitioning to data engineering. dbt expert with Spark experience.",
        "education": "B.Tech IT",
    },
    {
        "candidate_id": "C008", "name": "Ananya Bose",
        "title": "Lead Data Platform Engineer",
        "skills": ["Python", "Spark", "Kafka", "Kubernetes", "Airflow", "Terraform", "AWS", "dbt", "Redshift"],
        "summary": "Lead data platform engineer. Built lakehouse on AWS. Mentors 6-person team. End-to-end ownership.",
        "education": "M.Sc Computer Science",
        "endorsements": 89,
    },
]

DEMO_JD = """Senior Data Engineer -- FinTech Platform

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


def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def _cache_load(cache_dir: Path, key: str):
    p = cache_dir / f"{key}.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def _cache_save(cache_dir: Path, key: str, data) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / f"{key}.pkl", "wb") as f:
        pickle.dump(data, f)


def _jd_to_embed_text(parsed_jd: ParsedJD) -> str:
    skills = [s.name for s in parsed_jd.explicit_skills + parsed_jd.implied_skills]
    return (
        f"Role: {parsed_jd.role_title}. "
        f"Seniority: {parsed_jd.seniority}. "
        f"Summary: {parsed_jd.summary} "
        f"Skills: {', '.join(skills)}."
    )


def load_profiles_auto(data_dir: Path) -> List[Dict]:
    json_p = data_dir / "profiles.json"
    csv_p  = data_dir / "profiles.csv"
    if json_p.exists() or csv_p.exists():
        return load_profiles(data_dir)
    print("[agent] No profiles found — using built-in demo profiles (8 candidates).")
    return DEMO_PROFILES


def run_pipeline(
    jd_text:  str,
    profiles: List[Dict] = None,
    rerank_n: int = None,
    out_dir:  Path = None,
    force:    bool = False,
) -> List[Dict]:
    """Run all 5 stages end-to-end. Returns final ranked list."""
    from sentence_transformers import SentenceTransformer

    data_dir  = Path(out_dir or "data")
    cache_dir = data_dir / ".cache"
    rerank_n  = rerank_n or int(os.getenv("PARAKH_RERANK_N", RERANK_N))
    jd_hash   = _md5(jd_text)

    # Stage 1: JD parsing
    s1_key    = f"parsed_jd_{jd_hash}"
    parsed_jd = None if force else _cache_load(cache_dir, s1_key)
    if parsed_jd is None:
        model_tag = os.getenv("PARAKH_MODEL", "llama3.2")
        print(f"[Stage 1] Parsing JD with {os.getenv('LLM_PROVIDER','ollama')}:{model_tag}...")
        parsed_jd = parse_jd(jd_text)
        _cache_save(cache_dir, s1_key, parsed_jd)
        save_parsed_jd(parsed_jd, data_dir / "parsed_jd.json")
    else:
        print("[Stage 1] Loaded parsed JD from cache.")

    # Stage 2: Recall
    if profiles is None:
        profiles = load_profiles_auto(data_dir)
    profiles_hash = _md5(json.dumps(
        [p.get("candidate_id", str(i)) for i, p in enumerate(profiles)]
    ))
    s2_key   = f"recall_{jd_hash}_{profiles_hash}"
    recalled = None if force else _cache_load(cache_dir, s2_key)
    if recalled is None:
        print(f"[Stage 2] Embedding {len(profiles)} profiles...")
        model         = SentenceTransformer(os.getenv("PARAKH_EMBED_MODEL", EMBED_MODEL))
        profile_texts = [profile_to_text(p) for p in profiles]
        jd_vec        = model.encode(
            [_jd_to_embed_text(parsed_jd)],
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype("float32")
        cand_vecs = embed_texts(profile_texts, model)
        index     = build_index(cand_vecs)
        k         = min(int(os.getenv("PARAKH_RECALL_K", RECALL_K)), len(profiles))
        recalled  = recall_top_k(jd_vec, index, profiles, k=k)
        _cache_save(cache_dir, s2_key, recalled)
        print(f"[Stage 2] Recalled {len(recalled)} candidates.")
    else:
        print(f"[Stage 2] Loaded {len(recalled)} recalled candidates from cache.")

    # Stage 3: Scoring
    s3_key = f"scored_{s2_key}"
    scored = None if force else _cache_load(cache_dir, s3_key)
    if scored is None:
        print("[Stage 3] Multi-signal scoring...")
        scored = score_candidates(recalled, parsed_jd)
        _cache_save(cache_dir, s3_key, scored)
        save_scored(scored, data_dir / "scored.json")
    else:
        print(f"[Stage 3] Loaded {len(scored)} scored candidates from cache.")

    top_n          = int(os.getenv("PARAKH_SCORE_N", SCORE_N))
    top_candidates = scored[:top_n]

    # Stage 4: LLM rerank
    s4_key = f"ranked_{s3_key}_{rerank_n}"
    ranked = None if force else _cache_load(cache_dir, s4_key)
    if ranked is None:
        print(f"[Stage 4] LLM rerank (top {rerank_n})...")
        ranked = rerank_candidates(top_candidates, parsed_jd, top_n=rerank_n)
        _cache_save(cache_dir, s4_key, ranked)
    else:
        print(f"[Stage 4] Loaded {len(ranked)} reranked candidates from cache.")

    # Stage 5: Output
    normalize_scores(ranked)
    write_output(ranked, out_dir=data_dir)
    print_summary(ranked)

    return ranked


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="PARAKH pipeline")
    ap.add_argument("jd_file",  nargs="?", help="Path to job description text file")
    ap.add_argument("rerank_n", nargs="?", type=int, default=None)
    ap.add_argument("--force",  action="store_true", help="Bypass all caches")
    args = ap.parse_args()

    data_dir = Path(__file__).parent.parent / "data"

    if args.jd_file:
        jd_text = load_jd(args.jd_file)
    else:
        jd_path = data_dir / "job_description.txt"
        if jd_path.exists():
            jd_text = load_jd(jd_path)
            print(f"[agent] Using JD from {jd_path}")
        else:
            jd_text = DEMO_JD
            print("[agent] No JD file found — using built-in demo JD.")

    t0 = time.time()
    ranked = run_pipeline(
        jd_text,
        out_dir=data_dir,
        rerank_n=args.rerank_n,
        force=args.force,
    )
    print(f"\n[agent] Done in {time.time()-t0:.1f}s  ({len(ranked)} candidates ranked)")
