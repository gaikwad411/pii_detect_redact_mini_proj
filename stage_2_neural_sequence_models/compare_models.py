"""
compare_models.py
-----------------
Reads results/summary.json and prints the full Stage 2 progression table.

Usage
-----
    python compare_models.py
    python compare_models.py --summary results/summary.json
"""

import json
import argparse
from pathlib import Path

MODEL_ORDER = ["RNN", "LSTM", "GRU", "BiLSTM", "BiLSTM+CRF"]

PROBLEMS = {
    "RNN":        "Vanishing gradients → can't learn long-range dependencies",
    "LSTM":       "Unidirectional → misses right context; 4× params vs RNN",
    "GRU":        "Unidirectional → misses right context (but 3× faster than LSTM)",
    "BiLSTM":     "Label independence → invalid BIO sequences possible",
    "BiLSTM+CRF": "✓ Best overall — motivates Stage 3 Transformers",
}

MOTIVATIONS = {
    "RNN":        "→ LSTM: add gating to fix vanishing gradients",
    "LSTM":       "→ GRU: fewer parameters, same quality; → BiLSTM: add right context",
    "GRU":        "→ BiLSTM: bidirectionality needed for NER context",
    "BiLSTM":     "→ BiLSTM+CRF: add explicit label transition constraints",
    "BiLSTM+CRF": "→ Stage 3: Transformers for subword + pre-training + global context",
}


def load_summary(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    # Keep only the latest run per model (last occurrence wins)
    seen = {}
    for row in rows:
        seen[row["model"]] = row
    ordered = [seen[m] for m in MODEL_ORDER if m in seen]
    return ordered


def print_main_table(rows: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("STAGE 2 — NEURAL NER PROGRESSION TABLE")
    print("=" * 90)
    header = f"{'Model':<16} {'Macro F1':>9} {'Recall':>8} {'PER Rec':>9} {'ORG Rec':>9} {'LOC Rec':>9} {'Inv BIO':>8}"
    print(header)
    print("-" * 90)

    prev_f1 = None
    for row in rows:
        pc    = row.get("per_class", {})
        per_r = pc.get("PER", {}).get("recall", 0.0)
        org_r = pc.get("ORG", {}).get("recall", 0.0)
        loc_r = pc.get("LOC", {}).get("recall", 0.0)
        f1    = row["macro_f1"]
        rec   = row["macro_recall"]
        inv   = row["invalid_bio"]

        delta = f"  (+{f1 - prev_f1:.4f})" if prev_f1 is not None else ""
        prev_f1 = f1

        print(
            f"{row['model']:<16} {f1:>9.4f} {rec:>8.4f} "
            f"{per_r:>9.4f} {org_r:>9.4f} {loc_r:>9.4f} {inv:>8}{delta}"
        )
    print("=" * 90)


def print_narrative_table(rows: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("PROBLEM → SOLUTION NARRATIVE ARC")
    print("=" * 90)
    print(f"{'Model':<16} {'F1':>7} {'Recall':>8}  {'Problem / Achievement'}")
    print("-" * 90)
    for row in rows:
        print(f"{row['model']:<16} {row['macro_f1']:>7.4f} {row['macro_recall']:>8.4f}  {PROBLEMS[row['model']]}")
        print(f"{'':16} {'':>7} {'':>8}  {MOTIVATIONS[row['model']]}")
        print()
    print("=" * 90)


def print_per_class_table(rows: list[dict]) -> None:
    print("\n" + "=" * 75)
    print("PER-CLASS RECALL PROGRESSION")
    print("=" * 75)
    print(f"{'Model':<16} {'PER':>10} {'ORG':>10} {'LOC':>10}  {'Dominant gap'}")
    print("-" * 75)

    prev = None
    for row in rows:
        pc    = row.get("per_class", {})
        per_r = pc.get("PER", {}).get("recall", 0.0)
        org_r = pc.get("ORG", {}).get("recall", 0.0)
        loc_r = pc.get("LOC", {}).get("recall", 0.0)

        if prev:
            d_per = per_r - prev["per"]
            d_org = org_r - prev["org"]
            d_loc = loc_r - prev["loc"]
            biggest = max([("PER", d_per), ("ORG", d_org), ("LOC", d_loc)], key=lambda x: x[1])
            gap = f"{biggest[0]} {'+' if biggest[1] >= 0 else ''}{biggest[1]:.4f}"
        else:
            gap = "baseline"

        print(f"{row['model']:<16} {per_r:>10.4f} {org_r:>10.4f} {loc_r:>10.4f}  {gap}")
        prev = {"per": per_r, "org": org_r, "loc": loc_r}
    print("=" * 75)


def print_invalid_bio_table(rows: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("INVALID BIO SEQUENCE COUNT  (target = 0)")
    print("=" * 60)
    max_inv = max(r["invalid_bio"] for r in rows) or 1
    bar_width = 30
    for row in rows:
        inv   = row["invalid_bio"]
        bar   = "█" * int(inv / max_inv * bar_width)
        print(f"  {row['model']:<14} {inv:>5}  {bar}")
    print("=" * 60)
    print("  BiLSTM+CRF achieves 0 via Viterbi decoding.")


def print_stage1_comparison(rows: list[dict]) -> None:
    stage1_f1     = 0.2183
    stage1_recall = 0.1591
    best = rows[-1]
    print("\n" + "=" * 60)
    print("STAGE 1 (Regex) vs STAGE 2 BEST (BiLSTM+CRF)")
    print("=" * 60)
    print(f"  {'':20} {'Macro F1':>10} {'Recall':>10}")
    print(f"  {'Stage 1 — Regex':<20} {stage1_f1:>10.4f} {stage1_recall:>10.4f}")
    print(f"  {'Stage 2 — BiLSTM+CRF':<20} {best['macro_f1']:>10.4f} {best['macro_recall']:>10.4f}")
    improvement_f1  = best["macro_f1"]     / stage1_f1
    improvement_rec = best["macro_recall"] / stage1_recall
    print(f"  {'Improvement':<20} {improvement_f1:>9.1f}x {improvement_rec:>9.1f}x")
    print("=" * 60)


def print_simple_table(rows: list[dict]) -> None:
    STAGE1_F1     = 0.2183
    STAGE1_RECALL = 0.1591

    print("\n" + "=" * 70)
    print("RESULTS TABLE — Stage 1 + Stage 2")
    print("=" * 70)
    print(f"  {'Model':<22} {'Approach':<12} {'Macro F1':>9} {'Recall':>9} {'Inv BIO':>8}")
    print("-" * 70)
    print(f"  {'Regex (Stage 1)':<22} {'Rule-based':<12} {STAGE1_F1:>9.4f} {STAGE1_RECALL:>9.4f} {'N/A':>8}")
    print("-" * 70)
    for row in rows:
        inv = row.get("invalid_bio", "N/A")
        inv_str = str(inv) if inv != "N/A" else "N/A"
        print(f"  {row['model']:<22} {'Neural':<12} {row['macro_f1']:>9.4f} {row['macro_recall']:>9.4f} {inv_str:>8}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Stage 2 model comparison")
    parser.add_argument("--summary", default="results/summary.json",
                        help="Path to summary.json (default: results/summary.json)")
    parser.add_argument("--simple", action="store_true",
                        help="Print simple results table only (with Stage 1 regex)")
    args = parser.parse_args()

    path = Path(args.summary)
    if not path.exists():
        print(f"[ERROR] {path} not found. Run run_stage2.py first.")
        return

    rows = load_summary(path)
    if not rows:
        print("[ERROR] No model results found in summary.json.")
        return

    if args.simple:
        print_simple_table(rows)
        return

    print_main_table(rows)
    print_narrative_table(rows)
    print_per_class_table(rows)
    print_invalid_bio_table(rows)
    print_stage1_comparison(rows)


if __name__ == "__main__":
    main()
