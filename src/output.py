"""
output.py — Stage 5: Output Writer

Writes the final ranked shortlist in two formats:
  - ranked_output.csv   primary deliverable; organizers' format
  - ranked_output.json  full data for the Streamlit demo

Also prints a Rich-formatted leaderboard to the terminal.
"""

import csv
import json
from pathlib import Path
from typing import List, Dict

from config import ID_FIELDS

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box as rich_box
    _RICH = True
except ImportError:
    _RICH = False


CSV_COLUMNS = [
    "rank", "candidate_id", "score_100", "final_score", "llm_score", "confidence",
    "hidden_gem", "reason", "skill_score", "seniority_score",
    "activity_score", "composite_score", "embedding_score",
]


def _get_id(candidate: dict) -> str:
    for field in ID_FIELDS:
        val = candidate.get(field)
        if val:
            return str(val)
    return "UNKNOWN"


def _to_row(rank: int, candidate: dict) -> dict:
    return {
        "rank":             rank,
        "candidate_id":     _get_id(candidate),
        "score_100":        round(float(candidate.get("score_100",        0.0)), 1),
        "final_score":      round(float(candidate.get("final_score",      0.0)), 4),
        "llm_score":        candidate.get("llm_score",      "N/A"),
        "confidence":       candidate.get("confidence",     "N/A"),
        "hidden_gem":       "yes" if candidate.get("hidden_gem") else "no",
        "reason":           candidate.get("reason",         ""),
        "skill_score":      round(float(candidate.get("skill_score",      0.0)), 4),
        "seniority_score":  round(float(candidate.get("seniority_score",  0.0)), 4),
        "activity_score":   round(float(candidate.get("activity_score",   0.0)), 4),
        "composite_score":  round(float(candidate.get("composite_score",  0.0)), 4),
        "embedding_score":  round(float(candidate.get("embedding_score",  0.0)), 4),
    }


def write_output(
    ranked: List[Dict],
    out_dir: str | Path = "data",
    stem: str = "ranked_output",
) -> tuple[Path, Path]:
    """Write ranked shortlist to CSV and JSON. Returns (csv_path, json_path)."""
    out_dir   = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"

    rows = [_to_row(rank, c) for rank, c in enumerate(ranked, start=1)]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    enriched = []
    for rank, c in enumerate(ranked, start=1):
        entry = dict(c)
        entry["rank"]         = rank
        entry["candidate_id"] = _get_id(c)
        entry.pop("_id",        None)
        entry.pop("_rank_jump", None)
        enriched.append(entry)

    json_path.write_text(
        json.dumps(enriched, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print(f"[output] Wrote {len(rows)} candidates -> {csv_path}")
    print(f"[output] Wrote full data   -> {json_path}")
    return csv_path, json_path


def normalize_scores(ranked: List[Dict]) -> List[Dict]:
    """Add score_100 = final_score * 100 to each candidate (in-place)."""
    for c in ranked:
        c["score_100"] = round(c.get("final_score", 0.0) * 100, 1)
    return ranked


def _conf_color(conf: str) -> str:
    return {"high": "green", "medium": "yellow", "low": "red"}.get(
        str(conf).lower(), "white"
    )


def print_summary(ranked: List[Dict], top_n: int = 10) -> None:
    """Print a leaderboard — uses Rich if available, falls back to plain text."""
    if _RICH:
        _print_summary_rich(ranked, top_n)
    else:
        _print_summary_plain(ranked, top_n)


def _print_summary_rich(ranked: List[Dict], top_n: int) -> None:
    console = Console()
    gems    = [c for c in ranked if c.get("hidden_gem")]
    top     = ranked[:top_n]

    table = Table(
        title=f"PARAKH -- Final Shortlist  ({len(ranked)} candidates ranked)",
        box=rich_box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        border_style="cyan",
        highlight=True,
        expand=False,
    )
    table.add_column("#",      style="dim",    width=3,  justify="right")
    table.add_column("ID",                     width=12)
    table.add_column("Title",                  width=22)
    table.add_column("Score",  justify="right",width=6)
    table.add_column("LLM",   justify="right", width=4)
    table.add_column("Conf",  justify="center",width=7)
    table.add_column("Skill", justify="right", width=6)
    table.add_column("Gem",   justify="center",width=3)
    table.add_column("Reason",                 min_width=30, no_wrap=False)

    for i, c in enumerate(top, 1):
        is_gem    = c.get("hidden_gem", False)
        conf      = str(c.get("confidence", "-")).lower()
        score     = c.get("final_score", 0.0)
        row_style = "bold yellow" if is_gem else ("bold" if i <= 3 else "")

        table.add_row(
            str(i),
            _get_id(c)[:11],
            str(c.get("title", c.get("current_role", "")))[:21],
            f"{score:.3f}",
            str(c.get("llm_score", "-")),
            f"[{_conf_color(conf)}]{conf[:4]}[/]",
            f"{c.get('skill_score', 0):.0%}",
            "*" if is_gem else "",
            str(c.get("reason", ""))[:90],
            style=row_style,
        )

    console.print()
    console.print(table)
    console.print()
    _print_score_breakdown_rich(console, top[:5])

    if gems:
        gem_lines = "\n".join(
            f"  [bold yellow]*[/] [{_get_id(c)}] {c.get('title', '')} "
            f"-- {c.get('reason', '')[:70]}"
            for c in gems
        )
        console.print(
            Panel(
                gem_lines,
                title=f"[bold yellow]* Hidden Gems  ({len(gems)} found)[/]",
                border_style="yellow",
            )
        )
        console.print()


def _print_score_breakdown_rich(console, top5: List[Dict]) -> None:
    if not top5:
        return
    t = Table(
        title="Score Breakdown -- Top 5",
        box=rich_box.SIMPLE,
        header_style="bold",
    )
    t.add_column("ID",           width=14)
    t.add_column("Embed",        justify="right", width=7)
    t.add_column("Skill",        justify="right", width=7)
    t.add_column("Seniority",    justify="right", width=10)
    t.add_column("Activity",     justify="right", width=9)
    t.add_column(">> Composite", justify="right", width=12)

    for c in top5:
        t.add_row(
            _get_id(c)[:13],
            f"{c.get('embedding_score', 0):.3f}",
            f"{c.get('skill_score',     0):.3f}",
            f"{c.get('seniority_score', 0):.3f}",
            f"{c.get('activity_score',  0):.3f}",
            f"[bold]{c.get('composite_score', 0):.3f}[/]",
        )
    console.print(t)


def _print_summary_plain(ranked: List[Dict], top_n: int) -> None:
    gems = [c for c in ranked if c.get("hidden_gem")]
    top  = ranked[:top_n]

    print(f"\n{'='*70}")
    print(f"  PARAKH -- Final Shortlist  ({len(ranked)} candidates ranked)")
    print(f"{'='*70}")
    print(f"{'#':<4} {'ID':<14} {'Score':>6} {'LLM':>4} {'Conf':<8} {'Gem':<4} Reason")
    print(f"{'-'*70}")
    for i, c in enumerate(top, 1):
        gem    = "*" if c.get("hidden_gem") else ""
        cid    = _get_id(c)[:13]
        reason = str(c.get("reason", ""))[:45]
        score  = c.get("final_score", 0.0)
        llm    = c.get("llm_score", "-")
        conf   = str(c.get("confidence", "-"))[:7]
        print(f"{i:<4} {cid:<14} {score:>6.3f} {str(llm):>4} {conf:<8} {gem:<4} {reason}")

    if gems:
        print(f"\n  * Hidden Gems ({len(gems)} found):")
        for c in gems:
            title = str(c.get("title", c.get("current_role", "")))
            print(f"    [{_get_id(c)}] {title} -- {str(c.get('reason', ''))[:60]}")
    print(f"{'='*70}\n")
