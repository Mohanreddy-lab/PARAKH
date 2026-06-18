# PARAKH — Intelligent Candidate Ranking

> Built for the **India Runs Data & AI Challenge — Track 1**

PARAKH ranks job candidates against a job description. It goes beyond keyword
matching: it understands what the role actually needs, scores candidates on
multiple signals, and explains every ranking decision in plain English — using
only evidence that exists in the candidate's profile.

---

## Why "PARAKH"?

*Parakh* (परख) means **to assess** or **to evaluate carefully** in Hindi. The
name captures the goal: thoughtful, evidence-based evaluation — not a keyword
scanner, not a black box.

---

## How it works — 5 stages

### Stage 1 · JD Intelligence (`src/jd_parser.py`)
Parse the job description with GPT-4o via LangChain. Extract:
- **Explicit skills** — things the JD directly mentions.
- **Implied skills** — things the role clearly needs but didn't spell out.
- **Seniority level** — junior / mid / senior, inferred from context.

Output is a structured dict used by every later stage.

### Stage 2 · Fast Recall (`src/recall.py`)
Embed the JD summary and every candidate profile with `sentence-transformers`.
Index all profiles in FAISS and do a nearest-neighbour search to pull the
**top ~200 candidates** for deeper analysis. This keeps the expensive LLM steps
fast and cheap.

### Stage 3 · Multi-Signal Scoring (`src/scoring.py`)
Score the recalled 200 on three signals, then fuse them:
| Signal | What it measures |
|---|---|
| Embedding similarity | Semantic closeness to the JD |
| Skill overlap | Weighted by how critical each skill is |
| Behavior signals | Activity/engagement data in the dataset (if available) |

`scikit-learn` handles the score normalisation and weighted fusion.

### Stage 4 · Honest LLM Rerank (`src/rerank.py`)
Take the **top ~50** from Stage 3 and ask GPT-4o to re-score each one.
Rules the LLM must follow:
- Quote real text from the profile as evidence.
- If evidence is thin, say so explicitly.
- Never invent or infer facts not present in the data.

Output: a final score (0–100) + a one-paragraph plain-English reason per candidate.

### Stage 5 · Output (`src/output.py`)
Write the ranked list in the organizers' exact required format. Each row includes:
- `rank`, `candidate_id`, `final_score`
- `reason` — the LLM explanation
- `hidden_gem` flag — `True` for candidates who scored high here but were ranked
  low by naive keyword matching.

---

## Project structure

```
PARAKH/
├── data/               # Dataset goes here (not committed to git)
├── src/
│   ├── jd_parser.py    # Stage 1 — JD Intelligence
│   ├── recall.py       # Stage 2 — Fast Recall (FAISS)
│   ├── scoring.py      # Stage 3 — Multi-Signal Scoring
│   ├── rerank.py       # Stage 4 — Honest LLM Rerank
│   ├── output.py       # Stage 5 — Output writer
│   └── demo.py         # Streamlit demo UI
├── README.md
├── requirements.txt
└── CLAUDE.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."

# Linux / macOS
export OPENAI_API_KEY="sk-..."
```

Never put the key directly in code.

### 3. Add your dataset

Place the candidate profiles and job description file(s) inside `data/`.
Adjust the data loader in `recall.py` and `output.py` to match the
organizers' exact file format once the dataset is available.

### 4. Run the pipeline

```bash
python src/jd_parser.py   # Stage 1 — parse JD
python src/recall.py      # Stage 2 — build index, recall top 200
python src/scoring.py     # Stage 3 — score and fuse signals
python src/rerank.py      # Stage 4 — LLM deep evaluation
python src/output.py      # Stage 5 — write final ranked file
```

### 5. Run the demo UI (optional)

```bash
streamlit run src/demo.py
```

---

## Design decisions

- **No knowledge graph** — adds complexity without clear payoff for this task.
- **No career trajectory prediction** — we score present fit, not future potential.
- **No invented numbers** — every score comes from real data or the LLM's
  reasoning over real profile text. If the evidence isn't there, we say so.
- **FAISS for recall, LLM for rerank** — calling the LLM on thousands of
  candidates is too slow and expensive. FAISS narrows the field cheaply; the
  LLM spends its budget only on the most promising candidates.

---

## What we deliver

1. This GitHub repo with working, documented code.
2. This README explaining the design and key decisions.
3. A ranked output file in the exact format the organizers specify.
