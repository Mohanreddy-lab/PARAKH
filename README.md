# PARAKH — Intelligent Candidate Ranking

> Built for the **India Runs Data & AI Challenge — Track 1**

*Parakh* (परख) means **to assess carefully** in Hindi.  
PARAKH ranks job candidates not by keyword count but by understanding what the role truly needs, scoring across multiple signals, and explaining every decision with evidence from the actual profile.

**Runs 100% offline and free** — uses a local Ollama LLM. No paid API required. Candidate data never leaves the machine.

---

## What it delivers

1. A clean GitHub repo with working, documented code.
2. This README explaining the design and key decisions.
3. `ranked_output.csv` — the final ranked list in the organizers' format, with `score_100`, `reason`, and `hidden_gem` flag per candidate.

---

## How it works — 5 stages

### Stage 1 · JD Intelligence (`src/jd_parser.py`)
Sends the job description to the local LLM. Extracts:
- **Required skills** — what the JD explicitly demands.
- **Implied skills** — things the role clearly needs but the JD never mentioned (e.g. a "Senior Data Engineer" role implies Git, Linux, code review).
- **Seniority** — `junior / mid / senior / lead`, inferred from context.
- **Latent needs** — what the role truly tests ("owns pipelines end-to-end", "works under ambiguity").
- **Summary** — one plain-English sentence for a recruiter in a hurry.

### Stage 2 · Fast Recall (`src/recall.py`)
Encodes the JD and every candidate profile with `sentence-transformers` (`all-MiniLM-L6-v2`).  
Indexes all profiles in FAISS and does a nearest-neighbour search to pull the **top ~200 candidates** cheaply, before the slower scoring and LLM stages.

### Stage 3 · Multi-Signal Scoring (`src/scoring.py`)
Scores the recalled 200 on **4 signals**, then blends them:

| Signal | Weight | What it measures |
|---|---|---|
| Embedding similarity | 30% | Semantic closeness to the JD |
| Skill overlap | 40% | Synonym-aware weighted match (via `skills.py`) |
| Seniority match | 15% | Candidate level vs. JD requirement |
| Activity / behavior | 15% | Engagement signals (endorsements, GitHub stars, etc.) |

Weights are env-var tunable and auto-normalize. Skill matching uses a 60+ synonym dictionary so "PySpark" matches "Apache Spark", "AWS" matches "Amazon Web Services", etc.

**Hidden gem detection:** a candidate who ranks significantly better by composite score than by raw embedding score (rank-jump ≥ 2 positions) is flagged as a hidden gem — someone a keyword scanner would have buried.

### Stage 4 · Honest LLM Rerank (`src/rerank.py`)
Takes the **top ~50** from Stage 3 and asks the local LLM to re-score each one.

Rules the LLM must follow:
- Quote real text from the profile as evidence.
- If evidence is thin, say "Limited evidence:" explicitly.
- Never invent or infer facts not in the data.

Confidence weighting: a `low`-confidence LLM score drifts toward the Stage-3 composite (the safe signal) rather than overriding it.

### Stage 5 · Output (`src/output.py`)
Writes two files:
- `ranked_output.csv` — organizers' format, one row per candidate.
- `ranked_output.json` — full data for the Streamlit demo.

Prints a Rich-formatted terminal leaderboard with color-coded confidence and a hidden gem panel.

---

## Project structure

```
PARAKH/
├── data/                    # Dataset goes here (gitignored)
├── src/
│   ├── llm.py               # get_llm() — Ollama by default, Gemini fallback
│   ├── config.py            # All weights, thresholds, field lists
│   ├── skills.py            # 60+ skill synonym groups, word-boundary matching
│   ├── utils.py             # safe_parse_json, as_list
│   ├── jd_parser.py         # Stage 1 — JD Intelligence
│   ├── recall.py            # Stage 2 — Fast Recall (FAISS)
│   ├── scoring.py           # Stage 3 — Multi-Signal Scoring
│   ├── rerank.py            # Stage 4 — Honest LLM Rerank
│   ├── output.py            # Stage 5 — Output writer (CSV + JSON + Rich)
│   ├── agent.py             # Orchestrator — runs all 5 stages with caching
│   ├── demo.py              # Streamlit demo UI
│   └── api.py               # FastAPI REST endpoint
├── tests/                   # 73 tests, all passing
├── .env.example             # All PARAKH_* env vars documented
├── requirements.txt
└── CLAUDE.md
```

---

## Setup

### 1. Install Ollama

Download from [ollama.com](https://ollama.com) and pull a model:

```bash
ollama pull mistral:7b    # recommended — fast, good JSON output
# or
ollama pull phi3:mini     # lighter, faster
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure (optional)

Copy `.env.example` to `.env` and edit. Defaults work out of the box with `mistral:7b`.

```env
PARAKH_MODEL=mistral:7b
LLM_PROVIDER=ollama
```

---

## Running

### Full pipeline (recommended)

```powershell
# Windows PowerShell
$env:PARAKH_MODEL = "mistral:7b"
python src/agent.py
```

```bash
# Linux / macOS
PARAKH_MODEL=mistral:7b python src/agent.py
```

The agent runs all 5 stages with smart caching — re-runs skip unchanged stages instantly.

Use `--force` to bypass all caches:

```bash
python src/agent.py --force
```

### Stage by stage (development)

```bash
python src/jd_parser.py   # Stage 1 — parse JD, smoke test
python src/recall.py      # Stage 2 — build FAISS index, recall top-200
python src/scoring.py     # Stage 3 — 4-signal composite scoring
python src/rerank.py      # Stage 4 — LLM deep evaluation
```

### Streamlit demo UI

```bash
streamlit run src/demo.py
```

### REST API

```bash
uvicorn src.api:app --reload
# POST http://localhost:8000/api/v1/rank
# GET  http://localhost:8000/api/v1/models
# GET  http://localhost:8000/api/v1/synonyms
```

### Tests (no API key needed — LLM is mocked)

```bash
python -m pytest tests/ -v
```

---

## Real dataset

When the organizers' dataset arrives:

1. Place `profiles.json` or `profiles.csv` in `data/`.
2. Place `job_description.txt` in `data/`.
3. Run `python src/agent.py`.

The loader in `recall.py` handles both JSON and CSV automatically. Adjust `profile_to_text()` in `recall.py` if the field names differ from the defaults.

---

## Design decisions

| Decision | Reason |
|---|---|
| **Local Ollama, not OpenAI** | Free, offline, private — no API cost, no data leaving the machine |
| **FAISS for recall, LLM for rerank** | Calling LLM on thousands of candidates is too slow. FAISS narrows the field cheaply; the LLM spends its budget on the top candidates only |
| **Synonym-aware skill matching** | "PySpark" must match "Apache Spark". Raw string matching fails on real JDs |
| **Confidence weighting** | Low-confidence LLM scores drift toward the composite signal, not zero. Prevents a single failed API call from tanking a strong candidate |
| **Rank-jump hidden gems** | Candidates whose multi-signal composite rank is much better than their raw embedding rank reveal what keyword scanners miss |
| **No knowledge graph / trajectory** | Adds complexity without clear payoff for this task. We win on quality and honesty |
| **Stage caching** | Each stage's output is cached by an MD5 hash of its inputs. Development iteration is instant after the first run |

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` or `gemini` |
| `PARAKH_MODEL` | `llama3.2` | Model name (e.g. `mistral:7b`) |
| `PARAKH_EMBED_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `PARAKH_RECALL_K` | `200` | Candidates retrieved by FAISS |
| `PARAKH_SCORE_N` | `50` | Candidates passed to LLM rerank |
| `PARAKH_RERANK_N` | `50` | Candidates actually scored by LLM |
| `PARAKH_W_EMBED` | `0.30` | Embedding signal weight |
| `PARAKH_W_SKILL` | `0.40` | Skill overlap signal weight |
| `PARAKH_W_SENIORITY` | `0.15` | Seniority signal weight |
| `PARAKH_W_ACTIVITY` | `0.15` | Activity signal weight |
| `PARAKH_BLEND_COMPOSITE` | `0.40` | Composite weight in final blend |
| `PARAKH_BLEND_LLM` | `0.60` | LLM weight in final blend |
