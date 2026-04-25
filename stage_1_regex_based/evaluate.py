"""
evaluate.py
-----------
Evaluation for the rule-based (and later, model-based) NER system.

Uses the `seqeval` library which computes entity-level (span-level)
precision, recall and F1 — the standard for NER evaluation.

Entity-level means:
  - An entity is correct ONLY if every token in the span has the right tag.
  - Getting B-PER right but missing I-PER is counted as a full miss.

This is stricter than token-level accuracy, and is what your professor
will expect to see.

Install: pip install seqeval
"""

from seqeval.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from seqeval.scheme import IOB2


def evaluate(
    gold_sequences:      list[list[str]],
    predicted_sequences: list[list[str]],
    verbose: bool = True,
) -> dict:
    """
    Compute entity-level precision, recall, F1 for all entity types.

    Parameters
    ----------
    gold_sequences      : list of gold BIO tag lists (one per sentence)
    predicted_sequences : list of predicted BIO tag lists (one per sentence)
    verbose             : if True, print the full classification report

    Returns
    -------
    dict with keys:
        precision   : macro-averaged precision
        recall      : macro-averaged recall  ← optimise this for PII
        f1          : macro-averaged F1
        report      : full per-class report string
    """
    # seqeval expects list[list[str]]
    precision = precision_score(gold_sequences, predicted_sequences,
                                average="macro", scheme=IOB2,
                                zero_division=0)
    recall    = recall_score(gold_sequences, predicted_sequences,
                             average="macro", scheme=IOB2,
                             zero_division=0)
    f1        = f1_score(gold_sequences, predicted_sequences,
                         average="macro", scheme=IOB2,
                         zero_division=0)
    report    = classification_report(gold_sequences, predicted_sequences,
                                      scheme=IOB2, zero_division=0)

    if verbose:
        print("\n" + "="*60)
        print("ENTITY-LEVEL EVALUATION REPORT")
        print("="*60)
        print(report)
        print(f"Macro Precision : {precision:.4f}")
        print(f"Macro Recall    : {recall:.4f}   ← primary metric for PII")
        print(f"Macro F1        : {f1:.4f}")
        print("="*60 + "\n")

    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "report":    report,
    }


def per_class_recall(
    gold_sequences:      list[list[str]],
    predicted_sequences: list[list[str]],
) -> dict[str, float]:
    """
    Returns recall broken down by entity type.
    Useful for checking: "are we missing PER more than ORG?"

    Returns dict: {"PER": 0.82, "ORG": 0.61, ...}
    """
    from seqeval.metrics import recall_score

    # seqeval doesn't directly expose per-class recall as a dict,
    # so we parse the classification report.
    report = classification_report(
        gold_sequences, predicted_sequences,
        scheme=IOB2, zero_division=0, output_dict=True
    )

    per_class = {}
    for key, val in report.items():
        # Keys like "PER", "ORG", "LOC" — skip "micro avg", "macro avg" etc.
        if isinstance(val, dict) and "recall" in val:
            per_class[key] = round(val["recall"], 4)

    return per_class


def confusion_analysis(
    gold_sequences:      list[list[str]],
    predicted_sequences: list[list[str]],
    n_examples: int = 5,
) -> None:
    """
    Print example false negatives (missed PII) and false positives.

    A false negative here = a gold entity span that was predicted as O.
    These are the dangerous misses for PII redaction.

    Parameters
    ----------
    gold_sequences      : gold BIO tag lists
    predicted_sequences : predicted BIO tag lists
    n_examples          : how many examples to print per error type
    """
    fn_examples = []  # false negatives (missed PII)
    fp_examples = []  # false positives (over-redaction)

    for gold_seq, pred_seq in zip(gold_sequences, predicted_sequences):
        for i, (g, p) in enumerate(zip(gold_seq, pred_seq)):
            if g != "O" and p == "O":
                fn_examples.append((g, p, i))
            elif g == "O" and p != "O":
                fp_examples.append((g, p, i))

    print(f"\nFalse Negatives (missed PII) — first {n_examples}:")
    for g, p, i in fn_examples[:n_examples]:
        print(f"  position {i:3d} | gold={g:<12} pred={p}")

    print(f"\nFalse Positives (over-redaction) — first {n_examples}:")
    for g, p, i in fp_examples[:n_examples]:
        print(f"  position {i:3d} | gold={g:<12} pred={p}")
