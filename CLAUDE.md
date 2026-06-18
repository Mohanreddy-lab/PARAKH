# PARAKH — Intelligent Candidate Ranking

## What this is
PARAKH ranks job candidates for a job description.
It does NOT just match keywords. It understands the real need,
ranks by evidence, and explains every choice in plain words.
Built for the "India Runs" Data & AI Challenge (Track 1).

## What we must deliver (3 things, scored)
1. A clean GitHub repo with working code.
2. A README that explains the design and our decisions.
3. A ranked output file in the EXACT format the organizers ask for.

## How it works (5 simple stages)
1. JD Intelligence: read the job description. Pull out skills needed,
   hidden/implied skills, and seniority. Save as structured data.
2. Fast recall: turn the job + all profiles into embeddings.
   Use FAISS to quickly grab the top ~200 closest candidates.
3. Multi-signal scoring: score those candidates by mixing
   - meaning match (embedding similarity)
   - skill overlap (weighted by how important each skill is)
   - any activity/behavior signals in the data.
4. Honest rerank: for the top ~50, ask the LLM to score fit and
   write a short reason, using REAL text from the profile.
   If proof is weak, it must say so. Never make up reasons.
5. Output: write the ranked list in the organizers' format, with
   score, reason, and a "hidden gem" flag for strong but overlooked people.

## Tech stack
- Python
- sentence-transformers (embeddings)
- FAISS (fast search)
- scikit-learn (scoring/fusion)
- LangChain + GPT-4o (JD parsing + rerank + explanations)
- Streamlit (simple demo UI)

## Project structure
parakh/
  data/            # dataset goes here (not added yet)
  src/
    jd_parser.py   # Stage 1
    recall.py      # Stage 2
    scoring.py     # Stage 3
    rerank.py      # Stage 4
    output.py      # Stage 5
    demo.py        # Streamlit app
  README.md
  requirements.txt
  CLAUDE.md

## Coding rules
- Keep functions small and clearly named.
- Add short comments in plain English.
- Build ONE stage at a time. Test it before moving on.
- After each working stage, make a git commit.
- No secret API keys in code. Read from environment.

## What we are NOT building (keep it lean, avoid risk)
- No knowledge graph.
- No "career trajectory" prediction.
- No fake "candidate digital twin" simulation.
- No invented money/value numbers.
We win on quality and honesty, not on flashy unfinished parts.

## Important note about the dataset
We do not have the dataset yet. Keep data loading flexible.
When the real dataset arrives, adjust the loader and the output
format to match the organizers' spec exactly.