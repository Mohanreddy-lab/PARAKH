# Stage 3 — Multi-Signal Scoring
# Combines embedding similarity, weighted skill overlap, and behavior signals
# into a single composite score for each recalled candidate.

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from jd_parser import ParsedJD, load_parsed_jd

# ── Signal weights (must sum to 1.0) ─────────────────────────────────────────
# Tune these once we see the real dataset and can evaluate quality.
WEIGHT_EMBED   = 0.40   # semantic closeness to JD
WEIGHT_SKILL   = 0.45   # weighted skill overlap
WEIGHT_BEHAVIOR = 0.15  # activity / engagement signals (if data has them)

# Importance → numeric weight for skill overlap calculation
IMPORTANCE_WEIGHT = {
    "required":     1.0,
    "preferred":    0.6,
    "nice-to-have": 0.3,
}

TOP_N_FOR_RERANK = 50   # how many to pass to Stage 4

# ── Skill overlap ─────────────────────────────────────────────────────────────

def _build_skill_map(parsed: ParsedJD) -> Dict[str, float]:
    """Return {skill_name: importance_weight} for all JD skills."""
    skill_map: Dict[str, float] = {}
    for s in parsed.explicit_skills + parsed.implied_skills:
        key = s.name.lower().strip()
        weight = IMPORTANCE_WEIGHT.get(s.importance, 0.3)
        # Keep the higher weight if the same skill appears in both lists
        skill_map[key] = max(skill_map.get(key, 0.0), weight)
    return skill_map

def _extract_candidate_skills(profile: Dict[str, Any]) -> List[str]:
    """Pull the candidate's skills out of their profile dict."""
    for field in ("skills", "skill_set", "technologies", "tech_stack"):
        val = profile.get(field)
        if val:
            if isinstance(val, list):
                return [str(v).lower().strip() for v in val]
            if isinstance(val, str):
                return [s.lower().strip() for s in val.replace(",", " ").split()]
    return []

def skill_overlap_score(
    profile: Dict[str, Any],
    skill_map: Dict[str, float],
) -> float:
    """Weighted Jaccard-style skill overlap score in [0, 1].

    Score = sum(weights of matched skills) / sum(all JD skill weights).
    A candidate who covers every required skill scores 1.0.
    """
    if not skill_map:
        return 0.0

    candidate_skills = set(_extract_candidate_skills(profile))
    total_weight = sum(skill_map.values())
    if total_weight == 0:
        return 0.0

    matched_weight = sum(
        w for skill, w in skill_map.items() if skill in candidate_skills
    )
    return matched_weight / total_weight

# ── Behavior signals ──────────────────────────────────────────────────────────

def behavior_score(profile: Dict[str, Any]) -> float:
    """Extract engagement / activity signals from the profile.

    Returns a raw float; will be normalised later with MinMaxScaler.
    Returns 0.0 if no behavior data is present.

    Adjust field names when the real dataset schema is known.
    """
    signals = []

    for field in ("endorsements", "recommendations", "profile_views",
                  "connections", "articles_published", "projects_count",
                  "github_stars", "contributions"):
        val = profile.get(field)
        if val is not None:
            try:
                signals.append(float(val))
            except (ValueError, TypeError):
                pass

    if not signals:
        return 0.0

    # Simple mean of available signals (all on different scales — scaler fixes this)
    return float(np.mean(signals))

# ── Normalisation ─────────────────────────────────────────────────────────────

def _normalise(values: List[float]) -> List[float]:
    """Scale a list of floats to [0, 1] using min-max normalisation."""
    arr = np.array(values, dtype=float).reshape(-1, 1)
    if arr.max() == arr.min():
        return [0.5] * len(values)   # all identical → neutral
    scaled = MinMaxScaler().fit_transform(arr).flatten()
    return scaled.tolist()

# ── Composite scoring ─────────────────────────────────────────────────────────

def score_candidates(
    recalled: List[Dict[str, Any]],
    parsed_jd: ParsedJD,
) -> List[Dict[str, Any]]:
    """Compute a composite score for every recalled candidate.

    Returns the list sorted by composite_score descending.
    """
    skill_map = _build_skill_map(parsed_jd)

    # Compute raw signals
    embed_raw    = [c["_embed_score"]           for c in recalled]
    skill_raw    = [skill_overlap_score(c, skill_map) for c in recalled]
    behavior_raw = [behavior_score(c)            for c in recalled]

    # Normalise each signal independently
    embed_norm    = _normalise(embed_raw)
    skill_norm    = _normalise(skill_raw)
    behavior_norm = _normalise(behavior_raw)

    scored = []
    for i, candidate in enumerate(recalled):
        e = embed_norm[i]
        s = skill_norm[i]
        b = behavior_norm[i]

        # If dataset has no behavior data, redistribute its weight
        has_behavior = behavior_raw[i] != 0.0
        if not has_behavior:
            e_w = WEIGHT_EMBED   / (WEIGHT_EMBED + WEIGHT_SKILL)
            s_w = WEIGHT_SKILL   / (WEIGHT_EMBED + WEIGHT_SKILL)
            composite = e_w * e + s_w * s
        else:
            composite = WEIGHT_EMBED * e + WEIGHT_SKILL * s + WEIGHT_BEHAVIOR * b

        entry = dict(candidate)
        entry["_skill_score"]     = round(skill_raw[i], 4)
        entry["_skill_norm"]      = round(s, 4)
        entry["_embed_norm"]      = round(e, 4)
        entry["_behavior_norm"]   = round(b, 4)
        entry["_composite_score"] = round(composite, 4)
        scored.append(entry)

    scored.sort(key=lambda x: x["_composite_score"], reverse=True)

    # Assign Stage-3 rank
    for rank, c in enumerate(scored, start=1):
        c["_score_rank"] = rank

    return scored

# ── Save / load ───────────────────────────────────────────────────────────────

def save_scored(candidates: List[Dict[str, Any]], path: Path) -> None:
    path.write_text(
        json.dumps(candidates, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"Saved {len(candidates)} scored candidates → {path}")

def load_scored(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))

# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_dir     = Path(__file__).parent.parent / "data"
    jd_path      = data_dir / "parsed_jd.json"
    recall_path  = data_dir / "recall.json"
    scored_path  = data_dir / "scored.json"

    # Guards
    for p, label in [(jd_path, "parsed_jd.json"), (recall_path, "recall.json")]:
        if not p.exists():
            stage = "jd_parser.py" if "jd" in str(p) else "recall.py"
            print(f"{label} not found. Run {stage} first.")
            sys.exit(1)

    parsed_jd = load_parsed_jd(jd_path)
    print(f"JD: {parsed_jd.role_title} ({parsed_jd.seniority})")
    print(f"JD skills: {len(parsed_jd.explicit_skills)} explicit, "
          f"{len(parsed_jd.implied_skills)} implied")

    from recall import load_recall
    recalled = load_recall(recall_path)
    print(f"Recalled candidates: {len(recalled)}")

    print("\nScoring...")
    scored = score_candidates(recalled, parsed_jd)
    save_scored(scored, scored_path)

    # Preview top 10
    print(f"\n── Top {min(10, len(scored))} after multi-signal scoring ──────────────────")
    print(f"{'#':>3}  {'ID':<8}  {'Title':<28}  {'Embed':>6}  {'Skill':>6}  {'Total':>6}")
    print("─" * 65)
    for c in scored[:10]:
        cid   = c.get("candidate_id", c.get("id", "?"))
        title = c.get("title", c.get("current_role", ""))[:27]
        print(
            f"  {c['_score_rank']:>2}  {cid:<8}  {title:<28}  "
            f"{c['_embed_norm']:>6.3f}  {c['_skill_norm']:>6.3f}  "
            f"{c['_composite_score']:>6.3f}"
        )

    print(f"\nTop {TOP_N_FOR_RERANK} will be passed to Stage 4 (rerank.py).")
    top50 = scored[:TOP_N_FOR_RERANK]
    top50_path = data_dir / "top50.json"
    top50_path.write_text(
        json.dumps(top50, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"Saved top-{TOP_N_FOR_RERANK} → {top50_path}")
