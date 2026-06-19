"""
demo.py — PARAKH Streamlit Demo

Paste a job description, upload candidate profiles (or use demo data),
run the 5-stage pipeline, and see live results as GPT-4o scores each
candidate. Includes score breakdown charts and a CSV download.

Run: streamlit run src/demo.py
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PARAKH — Candidate Ranker",
    page_icon="P",
    layout="wide",
)

# ── Imports (after path setup) ────────────────────────────────────────────────

from config    import EMBED_MODEL, RECALL_K, SCORE_N, RERANK_N
from jd_parser import parse_jd, ParsedJD
from recall    import profile_to_text, embed_texts, build_index, recall_top_k
from scoring   import score_candidates
from rerank    import rerank_stream, _make_jd_summary, _score_one, _get_chain
from output    import normalize_scores, _get_id, _conf_color
from agent     import DEMO_PROFILES, DEMO_JD, _jd_to_embed_text


# ── Helpers ───────────────────────────────────────────────────────────────────

def _file_md5(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()[:12]


@st.cache_resource(show_spinner="Loading embedding model...")
def _get_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


@st.cache_data(show_spinner="Indexing profiles...")
def _build_faiss(profiles_hash: str, profile_texts: list, model_name: str):
    model     = _get_model(model_name)
    vecs      = embed_texts(profile_texts, model)
    index     = build_index(vecs)
    return index


def _confidence_badge(conf: str) -> str:
    icons = {"high": "green", "medium": "orange", "low": "red"}
    icon  = {"high": "circle-check", "medium": "circle", "low": "circle-x"}.get(conf, "circle")
    color = icons.get(conf, "gray")
    return f'<span style="color:{color}; font-weight:bold">{conf.upper()}</span>'


def _render_card(rank: int, c: dict) -> None:
    cid   = _get_id(c)
    title = str(c.get("title", c.get("current_role", "Unknown")))
    score = c.get("final_score", 0.0)
    gem   = c.get("hidden_gem", False)
    conf  = str(c.get("confidence", "low")).lower()

    header = f"**#{rank} — [{cid}] {title}**"
    if gem:
        header += "  ⭐ Hidden Gem"

    with st.expander(header, expanded=(rank <= 3)):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Final Score", f"{score:.3f}")
        col2.metric("LLM Score",   f"{c.get('llm_score', '-')}/10")
        col3.metric("Skill Match", f"{c.get('skill_score', 0):.0%}")
        col4.metric("Confidence",  conf.upper())

        st.write(f"**Reason:** {c.get('reason', '')}")

        ev = c.get("skill_evidence", {})
        if ev.get("required_matched"):
            st.success(f"Required matched: {', '.join(ev['required_matched'])}")
        if ev.get("required_missing"):
            st.error(f"Required missing: {', '.join(ev['required_missing'])}")
        if ev.get("preferred_matched"):
            st.info(f"Preferred matched: {', '.join(ev['preferred_matched'])}")

        cols = st.columns(4)
        cols[0].progress(c.get("embedding_score",  0.0), text=f"Embed {c.get('embedding_score', 0):.2f}")
        cols[1].progress(c.get("skill_score",      0.0), text=f"Skill {c.get('skill_score',     0):.2f}")
        cols[2].progress(c.get("seniority_score",  0.0), text=f"Seniority {c.get('seniority_score', 0):.2f}")
        cols[3].progress(c.get("activity_score",   0.0), text=f"Activity {c.get('activity_score',  0):.2f}")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("PARAKH Settings")

    model_name = st.text_input(
        "Ollama model",
        value=os.getenv("PARAKH_MODEL", "llama3.2"),
        help="Any model you have pulled: llama3.2, mistral:7b, etc.",
    )
    os.environ["PARAKH_MODEL"] = model_name

    st.divider()
    st.subheader("Signal Weights")
    w_embed     = st.slider("Embedding",  0.0, 1.0, 0.30, 0.05)
    w_skill     = st.slider("Skill",      0.0, 1.0, 0.40, 0.05)
    w_seniority = st.slider("Seniority",  0.0, 1.0, 0.15, 0.05)
    w_activity  = st.slider("Activity",   0.0, 1.0, 0.15, 0.05)

    os.environ["PARAKH_W_EMBED"]     = str(w_embed)
    os.environ["PARAKH_W_SKILL"]     = str(w_skill)
    os.environ["PARAKH_W_SENIORITY"] = str(w_seniority)
    os.environ["PARAKH_W_ACTIVITY"]  = str(w_activity)

    st.divider()
    rerank_n = st.number_input(
        "Candidates to rerank",
        min_value=1, max_value=100,
        value=int(os.getenv("PARAKH_RERANK_N", RERANK_N)),
    )

# ── Main ──────────────────────────────────────────────────────────────────────

st.title("PARAKH — Intelligent Candidate Ranking")
st.caption("Understands what the role really needs, ranks by evidence, explains every choice.")

tab_run, tab_jd, tab_results, tab_chart = st.tabs(["Run Pipeline", "Parsed JD", "Results", "Charts"])

# ── Tab 1: Inputs ─────────────────────────────────────────────────────────────

with tab_run:
    col_jd, col_pf = st.columns(2)

    with col_jd:
        st.subheader("Job Description")
        use_demo_jd = st.checkbox("Use demo JD", value=True)
        jd_text = st.text_area(
            "Paste JD here",
            value=DEMO_JD if use_demo_jd else "",
            height=280,
            disabled=use_demo_jd,
        )
        if use_demo_jd:
            jd_text = DEMO_JD

    with col_pf:
        st.subheader("Candidate Profiles")
        use_demo_pf = st.checkbox("Use demo profiles (8 candidates)", value=True)
        profiles = DEMO_PROFILES if use_demo_pf else None

        if not use_demo_pf:
            uploaded = st.file_uploader(
                "Upload profiles.json or profiles.csv",
                type=["json", "csv"],
            )
            if uploaded:
                raw = uploaded.read()
                if uploaded.name.endswith(".json"):
                    profiles = json.loads(raw)
                else:
                    import io
                    profiles = pd.read_csv(io.BytesIO(raw)).to_dict(orient="records")
                st.success(f"Loaded {len(profiles)} profiles.")

    run_btn = st.button(
        "Run Pipeline",
        type="primary",
        disabled=not jd_text or (not use_demo_pf and not profiles),
    )

# ── Pipeline execution ────────────────────────────────────────────────────────

if run_btn:
    if profiles is None:
        st.error("Upload profiles or use demo profiles.")
        st.stop()

    st.session_state["results"] = []
    st.session_state["parsed_jd"] = None

    with tab_run:
        progress_bar = st.progress(0, "Starting pipeline...")
        status_area  = st.empty()

        # Stage 1
        status_area.info("Stage 1/4: Parsing job description with GPT-4o...")
        try:
            parsed_jd = parse_jd(jd_text)
            st.session_state["parsed_jd"] = parsed_jd
            status_area.success(
                f"Stage 1 done: {parsed_jd.role_title} ({parsed_jd.seniority}) "
                f"— {len(parsed_jd.explicit_skills)} explicit skills, "
                f"{len(parsed_jd.implied_skills)} implied"
            )
        except Exception as e:
            st.error(f"Stage 1 failed: {e}")
            st.stop()
        progress_bar.progress(0.25)

        # Stage 2
        status_area.info("Stage 2/4: Embedding profiles and recalling top candidates...")
        model_name    = os.getenv("PARAKH_EMBED_MODEL", EMBED_MODEL)
        profile_texts = [profile_to_text(p) for p in profiles]
        model         = _get_model(model_name)
        jd_vec        = model.encode(
            [_jd_to_embed_text(parsed_jd)],
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype("float32")
        cand_vecs = embed_texts(profile_texts, model)
        index     = build_index(cand_vecs)
        k         = min(int(os.getenv("PARAKH_RECALL_K", RECALL_K)), len(profiles))
        recalled  = recall_top_k(jd_vec, index, profiles, k=k)
        status_area.success(f"Stage 2 done: recalled {len(recalled)} candidates.")
        progress_bar.progress(0.50)

        # Stage 3
        status_area.info("Stage 3/4: Multi-signal scoring...")
        scored        = score_candidates(recalled, parsed_jd)
        top_n         = int(os.getenv("PARAKH_SCORE_N", SCORE_N))
        top_candidates = scored[:top_n]
        status_area.success(f"Stage 3 done: scored {len(scored)} candidates.")
        progress_bar.progress(0.60)

        # Stage 4 — streaming
        model_tag = os.getenv("PARAKH_MODEL", "llama3.2")
        status_area.info(f"Stage 4/4: LLM reranking top {rerank_n} candidates (live, model={model_tag})...")
        results_area = st.empty()
        results      = []

        try:
            chain      = _get_chain()
            jd_summary = _make_jd_summary(parsed_jd)
            batch      = top_candidates[:rerank_n]

            for idx, candidate in enumerate(batch):
                scored_c = _score_one(candidate, chain, jd_summary)
                results.append(scored_c)
                results_sorted = sorted(results, key=lambda x: x["final_score"], reverse=True)
                normalize_scores(results_sorted)

                with results_area.container():
                    st.write(f"**Scored {idx+1}/{len(batch)}**")
                    for r_rank, r_c in enumerate(results_sorted[:5], 1):
                        cid = _get_id(r_c)
                        st.write(
                            f"  {r_rank}. [{cid}] {r_c.get('title','')}  "
                            f"score={r_c.get('final_score',0):.3f}  "
                            f"llm={r_c.get('llm_score','-')}/10  "
                            f"conf={r_c.get('confidence','-')}"
                        )

                progress_bar.progress(0.60 + 0.40 * (idx + 1) / len(batch))

        except Exception as e:
            st.error(f"Stage 4 failed: {e}")
            st.stop()

        final_ranked = sorted(results, key=lambda x: x["final_score"], reverse=True)
        normalize_scores(final_ranked)
        st.session_state["results"] = final_ranked
        progress_bar.progress(1.0)
        status_area.success(
            f"Pipeline complete! {len(final_ranked)} candidates ranked. "
            "See Results and Charts tabs."
        )

# ── Tab 2: Parsed JD ─────────────────────────────────────────────────────────

with tab_jd:
    parsed_jd_stored = st.session_state.get("parsed_jd")
    if not parsed_jd_stored:
        st.info("Run the pipeline first — the parsed JD will appear here.")
    else:
        p = parsed_jd_stored
        st.subheader(f"{p.role_title}  ·  {p.seniority.upper()}")
        st.caption(p.summary)

        col_r, col_i, col_l = st.columns(3)
        with col_r:
            st.markdown("**Required skills**")
            for s in p.explicit_skills:
                badge = ":red[required]" if s.importance == "required" else ":orange[preferred]"
                st.markdown(f"- {s.name}  {badge}")
        with col_i:
            st.markdown("**Implied skills**")
            for s in p.implied_skills:
                st.markdown(f"- {s.name}")
        with col_l:
            st.markdown("**Latent needs**")
            for n in p.latent_needs:
                st.markdown(f"- _{n}_")

        st.divider()
        with st.expander("Raw JSON"):
            st.json(p.model_dump())


# ── Tab 3: Results ────────────────────────────────────────────────────────────

with tab_results:
    results = st.session_state.get("results", [])
    if not results:
        st.info("Run the pipeline to see results.")
    else:
        gems = [c for c in results if c.get("hidden_gem")]
        if gems:
            st.warning(f"★ {len(gems)} hidden gem(s) found — see below!")

        for rank, c in enumerate(results, 1):
            _render_card(rank, c)

        # Download button
        rows = []
        for rank, c in enumerate(results, 1):
            rows.append({
                "rank": rank, "candidate_id": _get_id(c),
                "final_score": c.get("final_score", 0),
                "llm_score": c.get("llm_score", ""),
                "confidence": c.get("confidence", ""),
                "hidden_gem": "yes" if c.get("hidden_gem") else "no",
                "reason": c.get("reason", ""),
                "skill_score": c.get("skill_score", 0),
                "seniority_score": c.get("seniority_score", 0),
                "composite_score": c.get("composite_score", 0),
                "embedding_score": c.get("embedding_score", 0),
            })
        df = pd.DataFrame(rows)
        st.download_button(
            "Download ranked_output.csv",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="ranked_output.csv",
            mime="text/csv",
        )

# ── Tab 4: Charts ─────────────────────────────────────────────────────────────

with tab_chart:
    results = st.session_state.get("results", [])
    if not results:
        st.info("Run the pipeline to see charts.")
    else:
        try:
            import plotly.graph_objects as go

            top10 = results[:10]
            ids   = [_get_id(c) for c in top10]

            # Score breakdown stacked bar
            fig = go.Figure()
            for signal, field in [
                ("Embedding",  "embedding_score"),
                ("Skill",      "skill_score"),
                ("Seniority",  "seniority_score"),
                ("Activity",   "activity_score"),
            ]:
                fig.add_trace(go.Bar(
                    name=signal,
                    x=ids,
                    y=[c.get(field, 0) for c in top10],
                ))
            fig.update_layout(
                barmode="stack",
                title="Score Breakdown — Top 10",
                xaxis_title="Candidate",
                yaxis_title="Score",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Final score scatter with confidence color
            conf_colors = {"high": "#2ecc71", "medium": "#f39c12", "low": "#e74c3c"}
            fig2 = go.Figure()
            for c in top10:
                conf  = str(c.get("confidence", "low")).lower()
                color = conf_colors.get(conf, "#95a5a6")
                cid   = _get_id(c)
                fig2.add_trace(go.Scatter(
                    x=[c.get("composite_score", 0)],
                    y=[c.get("final_score", 0)],
                    mode="markers+text",
                    marker=dict(size=14, color=color),
                    text=[cid],
                    textposition="top center",
                    name=cid,
                    showlegend=False,
                ))
            fig2.update_layout(
                title="Composite vs Final Score (color = confidence)",
                xaxis_title="Composite Score (pre-LLM)",
                yaxis_title="Final Score (post-LLM)",
                height=400,
            )
            st.plotly_chart(fig2, use_container_width=True)

        except ImportError:
            st.info("Install plotly to see charts: pip install plotly")
