"""
scoring.py — Stage 3: Multi-Signal Scoring

4 signals: embedding similarity, synonym-aware skill overlap,
seniority match, and activity/behavior signals.

All signals normalized to [0,1] before blending. Weights are
env-var tunable and auto-normalized so they always sum to 1.

Hidden gems are candidates whose composite rank is significantly
better than their raw embedding rank (rank-jump heuristic).
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(__file__))

from jd_parser import ParsedJD, load_parsed_jd
from skills import skill_matches, matched_skills
from config import (
    get_weights, IMPORTANCE_WEIGHT, SCORE_N,
    SENIORITY_RANK, SENIOR_WORDS, LEAD_WORDS, MID_WORDS, JUNIOR_WORDS,
    SKILL_FIELDS, SENIORITY_TEXT_FIELDS, ACTIVITY_FIELDS,
    HIDDEN_GEM_MIN_COMPOSITE, HIDDEN_GEM_MIN_RANK_JUMP,
)


def _normalise(values: List[float]) -> List[float]:
    arr = np.array(values, dtype=float).reshape(-1, 1)
    if arr.max() == arr.min():
        return [0.5] * len(values)
    return MinMaxScaler().fit_transform(arr).flatten().tolist()


def _build_candidate_text(profile: Dict[str, Any]) -> str:
    parts = []
    for field in SENIORITY_TEXT_FIELDS:
        if profile.get(field):
            parts.append(str(profile[field]))
    for field in SKILL_FIELDS:
        val = profile.get(field)
        if val:
            if isinstance(val, list):
                val = " ".join(str(v) for v in val)
            parts.append(val)
    for field in ("experience", "work_experience", "summary", "bio", "about",
                  "education", "certifications", "github_repos", "projects"):
        if profile.get(field):
            parts.append(str(profile[field]))
    return " ".join(parts)


def _infer_seniority(profile: Dict[str, Any]) -> str:
    text = ""
    for field in SENIORITY_TEXT_FIELDS:
        text += " " + str(profile.get(field, "")).lower()
    words = set(re.findall(r'\b\w+\b', text))
    if words & LEAD_WORDS:    return "lead"
    if words & SENIOR_WORDS:  return "senior"
    if words & MID_WORDS:     return "mid"
    if words & JUNIOR_WORDS:  return "junior"
    return "mid"


def _seniority_score(candidate_level: str, jd_level: str) -> float:
    c = SENIORITY_RANK.get(candidate_level, 2)
    j = SENIORITY_RANK.get(jd_level, 2)
    if c >= j:     return 1.0
    if c == j - 1: return 0.55
    return 0.15


def _skill_score_with_evidence(
    profile: Dict[str, Any], parsed_jd: ParsedJD
) -> Tuple[float, dict]:
    candidate_text = _build_candidate_text(profile)

    total_weight   = 0.0
    matched_weight = 0.0
    all_skills = list(parsed_jd.explicit_skills) + list(parsed_jd.implied_skills)
    for s in all_skills:
        w = IMPORTANCE_WEIGHT.get(s.importance, 0.3)
        total_weight += w
        if skill_matches(s.name, candidate_text):
            matched_weight += w

    score = matched_weight / total_weight if total_weight > 0 else 0.0

    req  = [s.name for s in parsed_jd.explicit_skills if s.importance == "required"]
    pref = [s.name for s in parsed_jd.explicit_skills if s.importance != "required"]
    impl = [s.name for s in parsed_jd.implied_skills]
    req_m,  req_miss  = matched_skills(req,  candidate_text)
    pref_m, _         = matched_skills(pref, candidate_text)
    impl_m, _         = matched_skills(impl, candidate_text)

    return score, {
        "required_matched":  req_m,
        "required_missing":  req_miss,
        "preferred_matched": pref_m,
        "implied_matched":   impl_m,
    }


def _activity_score_raw(profile: Dict[str, Any]) -> float:
    signals = []
    for field in ACTIVITY_FIELDS:
        val = profile.get(field)
        if val is not None:
            try:
                signals.append(float(val))
            except (ValueError, TypeError):
                pass
    return float(np.mean(signals)) if signals else 0.0


def _detect_hidden_gems(scored: List[Dict[str, Any]]) -> None:
    embed_sorted = sorted(scored, key=lambda x: x["_embed_score"], reverse=True)
    embed_rank   = {id(c): i + 1 for i, c in enumerate(embed_sorted)}
    for i, c in enumerate(scored):
        comp_rank   = i + 1
        rank_jump   = embed_rank[id(c)] - comp_rank
        c["hidden_gem"] = (
            rank_jump >= HIDDEN_GEM_MIN_RANK_JUMP and
            c["composite_score"] >= HIDDEN_GEM_MIN_COMPOSITE
        )
        c["_rank_jump"] = rank_jump


def score_candidates(
    recalled: List[Dict[str, Any]],
    parsed_jd: ParsedJD,
) -> List[Dict[str, Any]]:
    """Compute 4-signal composite score for every recalled candidate."""
    weights  = get_weights()
    jd_level = (parsed_jd.seniority or "mid").lower()

    embed_raw      = [c["_embed_score"] for c in recalled]
    skill_data     = [_skill_score_with_evidence(c, parsed_jd) for c in recalled]
    skill_raw      = [d[0] for d in skill_data]
    seniority_raw  = [_seniority_score(_infer_seniority(c), jd_level) for c in recalled]
    activity_raw   = [_activity_score_raw(c) for c in recalled]

    embed_norm    = _normalise(embed_raw)
    activity_norm = _normalise(activity_raw)
    has_activity  = any(v > 0 for v in activity_raw)

    scored = []
    for i, candidate in enumerate(recalled):
        e = embed_norm[i]
        s = skill_raw[i]
        n = seniority_raw[i]
        a = activity_norm[i]

        if has_activity:
            composite = (weights["embed"] * e + weights["skill"] * s +
                         weights["seniority"] * n + weights["activity"] * a)
        else:
            w_total   = weights["embed"] + weights["skill"] + weights["seniority"]
            composite = (weights["embed"] / w_total * e +
                         weights["skill"] / w_total * s +
                         weights["seniority"] / w_total * n)

        entry = dict(candidate)
        entry["embedding_score"]  = round(e,         4)
        entry["skill_score"]      = round(s,         4)
        entry["seniority_score"]  = round(n,         4)
        entry["activity_score"]   = round(a,         4)
        entry["composite_score"]  = round(composite, 4)
        entry["skill_evidence"]   = skill_data[i][1]
        scored.append(entry)

    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    for rank, c in enumerate(scored, start=1):
        c["_score_rank"] = rank

    _detect_hidden_gems(scored)
    return scored


def save_scored(candidates: List[Dict[str, Any]], path: Path) -> None:
    Path(path).write_text(
        json.dumps(candidates, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"[scoring] Saved {len(candidates)} candidates -> {path}")


def load_scored(path: Path) -> List[Dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    data_dir    = Path(__file__).parent.parent / "data"
    jd_path     = data_dir / "parsed_jd.json"
    recall_path = data_dir / "recall.json"
    scored_path = data_dir / "scored.json"
    top50_path  = data_dir / "top50.json"

    for p, label in [(jd_path, "parsed_jd.json"), (recall_path, "recall.json")]:
        if not p.exists():
            stage = "jd_parser.py" if "jd" in str(p) else "recall.py"
            print(f"{label} not found. Run {stage} first.")
            import sys; sys.exit(1)

    parsed_jd = load_parsed_jd(jd_path)
    print(f"JD: {parsed_jd.role_title} ({parsed_jd.seniority})")

    from recall import load_recall
    recalled = load_recall(recall_path)
    print(f"Recalled: {len(recalled)} candidates")

    scored = score_candidates(recalled, parsed_jd)
    save_scored(scored, scored_path)

    top_n = SCORE_N
    print(f"\n-- Top {min(top_n, len(scored))} --")
    print(f"{'#':>3}  {'ID':<12}  {'Title':<24}  Embed  Skill  Snr  Total  Gem")
    print("-" * 72)
    for c in scored[:top_n]:
        cid   = str(c.get("candidate_id", c.get("id", "?")))[:12]
        title = str(c.get("title", c.get("current_role", "")))[:23]
        gem   = "*" if c.get("hidden_gem") else ""
        print(f"  {c['_score_rank']:>2}  {cid:<12}  {title:<24}  "
              f"{c['embedding_score']:>5.3f}  {c['skill_score']:>5.3f}  "
              f"{c['seniority_score']:>4.2f}  {c['composite_score']:>5.3f}  {gem}")

    top50 = scored[:top_n]
    top50_path.write_text(
        json.dumps(top50, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\nTop-{top_n} saved -> {top50_path}")
