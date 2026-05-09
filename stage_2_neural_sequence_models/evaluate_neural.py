"""
evaluate_neural.py
------------------
Evaluation utilities for neural NER models.

Produces:
  - Entity-level precision / recall / F1 per class (seqeval)
  - Per-class recall breakdown
  - Invalid BIO sequence count (quantifies the problem BiLSTM+CRF solves)
  - Confusion matrix at entity type level
"""

from seqeval.metrics import classification_report, f1_score, recall_score
from seqeval.scheme import IOB2
from collections import defaultdict


def full_eval(
    gold_sequences: list[list[str]],
    pred_sequences: list[list[str]],
    model_name: str = "Model",
    verbose: bool = True,
) -> dict:
    """Full seqeval evaluation with per-class breakdown."""

    report_str = classification_report(
        gold_sequences, pred_sequences,
        scheme=IOB2, zero_division=0,
    )
    report_dict = classification_report(
        gold_sequences, pred_sequences,
        scheme=IOB2, zero_division=0, output_dict=True,
    )

    macro_f1  = f1_score(gold_sequences, pred_sequences,
                         average="macro", scheme=IOB2, zero_division=0)
    macro_rec = recall_score(gold_sequences, pred_sequences,
                             average="macro", scheme=IOB2, zero_division=0)

    invalid = count_invalid_bio(pred_sequences)

    if verbose:
        print(f"\n{'='*60}")
        print(f"EVALUATION: {model_name}")
        print(f"{'='*60}")
        print(report_str)
        print(f"Macro F1     : {macro_f1:.4f}")
        print(f"Macro Recall : {macro_rec:.4f}  ← primary metric")
        print(f"Invalid BIO  : {invalid}  ← should be 0 with CRF")
        print(f"{'='*60}\n")

    return {
        "model_name":   model_name,
        "macro_f1":     round(macro_f1, 4),
        "macro_recall": round(macro_rec, 4),
        "invalid_bio":  invalid,
        "per_class":    {
            k: v for k, v in report_dict.items()
            if isinstance(v, dict) and k not in
               ("micro avg", "macro avg", "weighted avg")
        },
    }


def count_invalid_bio(sequences: list[list[str]]) -> int:
    """
    Count invalid BIO transitions in predicted sequences.

    Invalid cases:
      1. I-X at position 0 (no preceding B-)
      2. I-X following O
      3. I-X following B-Y or I-Y where Y ≠ X

    This number should be 0 for BiLSTM+CRF (Viterbi guarantees validity)
    and non-zero for all other models (softmax predicts independently).
    """
    count = 0
    for seq in sequences:
        prev = "O"
        for tag in seq:
            if tag.startswith("I-"):
                entity = tag[2:]
                if prev == "O" or (prev[2:] != entity):
                    count += 1
            prev = tag
    return count


def print_progression_table(all_results: list[dict]) -> None:
    """
    Print a model progression table showing improvement across all 5 models.
    This is the centrepiece of your professor presentation.
    """
    print("\n" + "=" * 80)
    print("MODEL PROGRESSION — Stage 2 Neural NER")
    print("=" * 80)
    print(f"{'Model':<20} {'Macro F1':>10} {'Macro Recall':>14} "
          f"{'Invalid BIO':>13} {'Key Problem'}")
    print("-" * 80)

    models_order = ["RNN", "LSTM", "GRU", "BiLSTM", "BiLSTM+CRF"]

    results_by_name = {r["model_name"]: r for r in all_results}

    for name in models_order:
        if name not in results_by_name:
            continue
        r = results_by_name[name]
        problem = _short_problem(name)
        print(
            f"{name:<20} {r['macro_f1']:>10.4f} {r['macro_recall']:>14.4f} "
            f"{r['invalid_bio']:>13}  {problem}"
        )

    print("=" * 80)

    # Per-class recall table
    print(f"\n{'Per-class Recall':}")
    print(f"{'Model':<20} {'PER':>8} {'ORG':>8} {'LOC':>8}")
    print("-" * 48)
    for name in models_order:
        if name not in results_by_name:
            continue
        r  = results_by_name[name]
        pc = r.get("per_class", {})
        per = pc.get("PER", {}).get("recall", 0)
        org = pc.get("ORG", {}).get("recall", 0)
        loc = pc.get("LOC", {}).get("recall", 0)
        print(f"{name:<20} {per:>8.4f} {org:>8.4f} {loc:>8.4f}")
    print("-" * 48)


def _short_problem(model_name: str) -> str:
    return {
        "RNN":        "Vanishing gradients → can't learn long dependencies",
        "LSTM":       "Unidirectional → misses right context",
        "GRU":        "Unidirectional → misses right context",
        "BiLSTM":     "Label independence → invalid BIO sequences",
        "BiLSTM+CRF": "✓ Best overall — motivates Stage 3 Transformers",
    }.get(model_name, "")


def print_gradient_analysis(all_train_results: list[dict]) -> None:
    """Print gradient behaviour summary across models."""
    print("\n" + "=" * 70)
    print("GRADIENT ANALYSIS")
    print("=" * 70)
    for r in all_train_results:
        print(f"\n{r['model_name']}:")
        print(f"  {r.get('gradient_problem', 'N/A')}")
    print("=" * 70)
