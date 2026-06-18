# Stage 2 — Fast Recall
# Converts the job description and all candidate profiles into embeddings.
# Uses FAISS to retrieve the top ~200 closest candidates quickly.

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from jd_parser import ParsedJD, load_parsed_jd

# ── Config ────────────────────────────────────────────────────────────────────

EMBED_MODEL = "all-MiniLM-L6-v2"   # fast, good quality, 384-dim
TOP_K = 200                         # candidates to recall for Stage 3

# ── Text builders ─────────────────────────────────────────────────────────────

def jd_to_text(parsed: ParsedJD) -> str:
    """Flatten a ParsedJD into a single string for embedding."""
    skills = [s.name for s in parsed.explicit_skills + parsed.implied_skills]
    return (
        f"Role: {parsed.role_title}. "
        f"Seniority: {parsed.seniority}. "
        f"Summary: {parsed.summary} "
        f"Skills: {', '.join(skills)}."
    )

def profile_to_text(profile: Dict[str, Any]) -> str:
    """Flatten a candidate profile dict into a single string for embedding.

    Adjust field names here once the real dataset schema is known.
    Currently handles both a generic dict and common field name variants.
    """
    parts = []

    # Try common field names; extend this list to match the real dataset
    for field in ("name", "title", "current_role", "role", "position"):
        if field in profile and profile[field]:
            parts.append(str(profile[field]))
            break

    for field in ("skills", "skill_set", "technologies", "tech_stack"):
        if field in profile and profile[field]:
            val = profile[field]
            if isinstance(val, list):
                val = ", ".join(str(v) for v in val)
            parts.append(f"Skills: {val}")
            break

    for field in ("experience", "work_experience", "summary", "bio", "about"):
        if field in profile and profile[field]:
            parts.append(str(profile[field]))
            break

    for field in ("education", "degree", "qualifications"):
        if field in profile and profile[field]:
            parts.append(str(profile[field]))
            break

    # Fallback: join all string values in the profile
    if not parts:
        parts = [str(v) for v in profile.values() if v and isinstance(v, str)]

    return " | ".join(parts)

# ── Data loader ───────────────────────────────────────────────────────────────

def load_profiles(data_dir: Path) -> List[Dict[str, Any]]:
    """Load candidate profiles from data_dir.

    Supports:
      - profiles.json  (list of dicts)
      - profiles.csv   (each row is a candidate)
    Extend here when the real dataset format is known.
    """
    json_path = data_dir / "profiles.json"
    csv_path  = data_dir / "profiles.csv"

    if json_path.exists():
        profiles = json.loads(json_path.read_text(encoding="utf-8"))
        print(f"Loaded {len(profiles)} profiles from {json_path.name}")
        return profiles

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        profiles = df.to_dict(orient="records")
        print(f"Loaded {len(profiles)} profiles from {csv_path.name}")
        return profiles

    raise FileNotFoundError(
        f"No profiles found in {data_dir}. "
        "Place profiles.json or profiles.csv there and re-run."
    )

# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_texts(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """Encode a list of strings → float32 numpy array (N, dim)."""
    vectors = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine sim via inner product
    )
    return vectors.astype("float32")

# ── FAISS index ───────────────────────────────────────────────────────────────

def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Build an exact inner-product (cosine) index over candidate vectors."""
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index

def save_index(index: faiss.Index, path: Path) -> None:
    faiss.write_index(index, str(path))

def load_index(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))

# ── Recall ────────────────────────────────────────────────────────────────────

def recall_top_k(
    jd_vector: np.ndarray,
    index: faiss.Index,
    profiles: List[Dict[str, Any]],
    k: int = TOP_K,
) -> List[Dict[str, Any]]:
    """Search the FAISS index and return top-k candidates with scores."""
    # jd_vector shape: (1, dim)
    scores, indices = index.search(jd_vector.reshape(1, -1), k)
    scores = scores[0]
    indices = indices[0]

    results = []
    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        if idx == -1:   # FAISS returns -1 when fewer than k items exist
            break
        candidate = dict(profiles[idx])
        candidate["_recall_rank"] = rank
        candidate["_embed_score"] = float(score)
        results.append(candidate)

    return results

# ── Save / load recall results ────────────────────────────────────────────────

def save_recall(results: List[Dict[str, Any]], path: Path) -> None:
    path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"Saved {len(results)} recalled candidates → {path}")

def load_recall(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))

# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_dir   = Path(__file__).parent.parent / "data"
    jd_path    = data_dir / "parsed_jd.json"
    index_path = data_dir / "faiss.index"
    meta_path  = data_dir / "profile_meta.pkl"
    recall_path = data_dir / "recall.json"

    # ── Guard: Stage 1 must have run first ───────────────────────────────────
    if not jd_path.exists():
        print(f"parsed_jd.json not found. Run jd_parser.py first.")
        sys.exit(1)

    parsed_jd = load_parsed_jd(jd_path)
    print(f"Loaded parsed JD: {parsed_jd.role_title} ({parsed_jd.seniority})")

    # ── Load profiles ─────────────────────────────────────────────────────────
    # If no real data yet, generate synthetic profiles for smoke-testing
    json_path = data_dir / "profiles.json"
    if not json_path.exists() and not (data_dir / "profiles.csv").exists():
        print("No profiles found — generating 50 synthetic profiles for testing.")
        import random, string
        random.seed(42)
        skill_pool = [
            "python", "sql", "spark", "aws", "dbt", "airflow", "kafka",
            "docker", "kubernetes", "java", "scala", "data modeling",
            "etl", "redshift", "snowflake", "pandas", "tensorflow", "ml pipelines",
        ]
        samples = []
        for i in range(50):
            k = random.randint(3, 8)
            samples.append({
                "candidate_id": f"C{i+1:03d}",
                "name": f"Candidate {i+1}",
                "title": random.choice(["Data Engineer", "ML Engineer",
                                        "Backend Engineer", "Analytics Engineer"]),
                "skills": random.sample(skill_pool, k),
                "experience": f"{random.randint(1,10)} years of experience in data and analytics.",
                "education": random.choice(["B.Tech CS", "M.Tech Data Science", "B.E. IT"]),
            })
        json_path.write_text(json.dumps(samples, indent=2), encoding="utf-8")
        print(f"Written synthetic profiles → {json_path}")

    profiles = load_profiles(data_dir)

    # ── Embed ─────────────────────────────────────────────────────────────────
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    jd_text = jd_to_text(parsed_jd)
    print(f"JD text for embedding:\n  {jd_text}\n")

    profile_texts = [profile_to_text(p) for p in profiles]

    print("Embedding candidates...")
    candidate_vectors = embed_texts(profile_texts, model)

    print("Embedding JD...")
    jd_vector = model.encode(
        [jd_text], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

    # ── Build & save index ────────────────────────────────────────────────────
    print("Building FAISS index...")
    index = build_index(candidate_vectors)
    save_index(index, index_path)

    # Save profile list so we can map FAISS index positions back to profiles
    with open(meta_path, "wb") as f:
        pickle.dump(profiles, f)
    print(f"Saved FAISS index → {index_path}")
    print(f"Saved profile meta → {meta_path}")

    # ── Recall ────────────────────────────────────────────────────────────────
    k = min(TOP_K, len(profiles))
    print(f"\nRecalling top {k} candidates...")
    recalled = recall_top_k(jd_vector, index, profiles, k=k)
    save_recall(recalled, recall_path)

    # Quick preview
    print("\n── Top 10 recalled ─────────────────────────────────────────────")
    for c in recalled[:10]:
        cid   = c.get("candidate_id", c.get("id", "?"))
        title = c.get("title", c.get("current_role", ""))
        score = c["_embed_score"]
        print(f"  #{c['_recall_rank']:>3}  {cid:<8}  {title:<30}  sim={score:.4f}")
