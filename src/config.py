"""
config.py — All magic numbers and field lists in one place.
"""

import os

def get_weights() -> dict:
    w_e = float(os.getenv("PARAKH_W_EMBED",     0.30))
    w_s = float(os.getenv("PARAKH_W_SKILL",     0.40))
    w_n = float(os.getenv("PARAKH_W_SENIORITY", 0.15))
    w_a = float(os.getenv("PARAKH_W_ACTIVITY",  0.15))
    total = w_e + w_s + w_n + w_a
    return {
        "embed":     w_e / total,
        "skill":     w_s / total,
        "seniority": w_n / total,
        "activity":  w_a / total,
    }

EMBED_MODEL = os.getenv("PARAKH_EMBED_MODEL", "all-MiniLM-L6-v2")
RECALL_K    = int(os.getenv("PARAKH_RECALL_K", 200))
SCORE_N     = int(os.getenv("PARAKH_SCORE_N",  50))
RERANK_N    = int(os.getenv("PARAKH_RERANK_N", 50))

MAX_PROFILE_CHARS = int(os.getenv("PARAKH_PROFILE_CHARS", 900))
BLEND_COMPOSITE   = float(os.getenv("PARAKH_BLEND_COMPOSITE", 0.40))
BLEND_LLM         = float(os.getenv("PARAKH_BLEND_LLM",       0.60))
LLM_RETRIES       = int(os.getenv("PARAKH_LLM_RETRIES",       2))

HIDDEN_GEM_MIN_COMPOSITE = float(os.getenv("PARAKH_HIDDEN_GEM_MIN_COMPOSITE", 0.40))
HIDDEN_GEM_MIN_RANK_JUMP = int(os.getenv("PARAKH_HIDDEN_GEM_MIN_RANK_JUMP",   2))

CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.70, "low": 0.30}

IMPORTANCE_WEIGHT = {
    "required":     1.0,
    "preferred":    0.6,
    "nice-to-have": 0.3,
}

SENIORITY_RANK = {"junior": 1, "mid": 2, "senior": 3, "lead": 4, "principal": 4}
SENIOR_WORDS   = {"senior", "sr", "principal", "staff", "distinguished"}
LEAD_WORDS     = {"lead", "head", "director", "vp", "chief", "manager", "architect"}
MID_WORDS      = {"mid", "ii", "iii", "intermediate"}
JUNIOR_WORDS   = {"junior", "jr", "associate", "entry", "intern", "graduate"}

SKILL_FIELDS          = ("skills", "skill_set", "technologies", "tech_stack")
SENIORITY_TEXT_FIELDS = ("title", "current_role", "role", "headline", "position")
ACTIVITY_FIELDS       = ("endorsements", "recommendations", "profile_views",
                         "connections", "articles_published", "projects_count",
                         "github_stars", "contributions")
ID_FIELDS             = ("candidate_id", "id", "profile_id", "email", "name")
