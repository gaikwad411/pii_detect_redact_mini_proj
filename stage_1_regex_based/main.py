"""
main.py
-------
Entry point for the Stage 1 rule-based Hindi PII detection system.

Expected project layout
-----------------------
    datasets/
        hi_IndicNER_v1.0/
            train.json          ← your dataset files
            test.json
    hindi_pii/                  ← this project (run scripts from here)
        main.py
        ...

Dataset format
--------------
Each file is a sequence of JSON objects separated only by whitespace
(no commas between objects, no enclosing array brackets).
Each object has the schema:
    {
        "words": ["token1", "token2", ...],
        "ner":   ["B-PER",  "O",      ...]
    }

Usage
-----
# Auto-detect dataset (looks one directory up for datasets/hi_IndicNER_v1.0/):
    python main.py

# Specify a file explicitly:
    python main.py --data ../datasets/hi_IndicNER_v1.0/train.json

# Limit rows for quick testing:
    python main.py --limit 500
"""

import json
import argparse
from pathlib import Path

from redactor  import process_example
from evaluate  import evaluate, per_class_recall, confusion_analysis


# ---------------------------------------------------------------------------
# Sample data  (your example + a few extras for testing)
# ---------------------------------------------------------------------------

SAMPLE_DATA = [
    {
        "words": [
            "गुजरात", "हाईकोर्ट", "ने", "मंगलवार", "को", "कांग्रेस", "नेता",
            "अहमद", "पटेल", "पर", "5", "हजार", "रुपये", "का", "जुर्माना",
            "लगा", "दिया", "जब", "उनके", "वकील", "ने", "2017", "में",
            "राज्यसभा", "चुनाव", "से", "पहले", "दिये", "गए", "विधायकों",
            "के", "बयानों", "की", "सीडी", "पेश", "करने", "के", "लिए",
            "एक", "गवाह", "को", "अनुमति", "देने", "की", "मांग", "की", "।"
        ],
        "ner": [
            "B-ORG", "I-ORG", "O", "O", "O", "B-ORG", "O",
            "B-PER", "I-PER", "O", "O", "O", "O", "O", "O",
            "O", "O", "O", "O", "O", "O", "O", "O",
            "B-ORG", "O", "O", "O", "O", "O", "O",
            "O", "O", "O", "O", "O", "O", "O", "O",
            "O", "O", "O", "O", "O", "O", "O", "O", "O"
        ]
    },
    {
        "words": [
            "डॉ", "मनमोहन", "सिंह", "ने", "दिल्ली", "में",
            "एक", "प्रेस", "कॉन्फ्रेंस", "की"
        ],
        "ner": [
            "O", "B-PER", "I-PER", "O", "B-LOC", "O",
            "O", "O", "O", "O"
        ]
    },
    {
        "words": [
            "उनका", "फोन", "नंबर", "9876543210", "है", "और",
            "आधार", "संख्या", "1234", "5678", "9012", "है"
        ],
        "ner": [
            "O", "O", "O", "B-PHONE", "O", "O",
            "O", "O", "B-AADHAAR", "I-AADHAAR", "I-AADHAAR", "O"
        ]
    },
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Default dataset location: one directory above this script
_HERE         = Path(__file__).parent
_DEFAULT_DIR  = _HERE.parent / "datasets" / "hi_IndicNER_v1.0"
# Prefer train split; fall back to any .json file found in the folder
_PREFERRED    = ["train.json", "train_NER_dataset.json", "data.json"]


def _find_default_dataset() -> Path | None:
    """
    Look for the dataset in the default location relative to this script.
    Returns the first matching file path, or None if nothing is found.
    """
    if not _DEFAULT_DIR.exists():
        return None
    for name in _PREFERRED:
        candidate = _DEFAULT_DIR / name
        if candidate.exists():
            return candidate
    # Fall back: any .json file in the directory
    json_files = sorted(_DEFAULT_DIR.glob("*.json"))
    return json_files[0] if json_files else None


def _iter_objects(text: str):
    """
    Generator that yields individual JSON objects from a string that
    contains multiple objects separated only by whitespace — i.e. NOT
    a JSON array and NOT comma-separated (NDJSON variant).

    Strategy: use a brace-depth counter to find object boundaries,
    then parse each slice independently. This is more robust than
    splitting on newlines when objects span multiple lines.
    """
    depth  = 0
    start  = None
    in_str = False
    escape = False

    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue

        if ch == "{":
            if depth == 0:
                start = i        # beginning of a new top-level object
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start : i + 1]
                start = None


def load_dataset(path: str | Path, limit: int | None = None) -> list[dict]:
    """
    Load examples from a JSON file whose objects are separated by
    whitespace only (no commas, no enclosing array).

    Also handles standard JSONL (one object per line) and a plain
    JSON array as fallback — so the same function works whatever
    format the file turns out to be.

    Parameters
    ----------
    path  : path to the .json / .jsonl file
    limit : maximum number of examples to return (None = all)

    Returns
    -------
    List of dicts with at least a "words" key.
    """
    path = Path(path)
    print(f"[INFO] Loading dataset from: {path}")

    raw = path.read_text(encoding="utf-8")

    # ── Try standard JSON array first (cheapest) ───────────────────────────
    stripped = raw.strip()
    if stripped.startswith("["):
        try:
            all_examples = json.loads(stripped)
            examples = all_examples[:limit] if limit else all_examples
            print(f"[INFO] Loaded {len(examples):,} examples (JSON array format)")
            return examples
        except json.JSONDecodeError:
            pass  # fall through to object-stream parser

    # ── Object-stream parser (your actual format) ──────────────────────────
    examples = []
    for obj_str in _iter_objects(raw):
        if limit and len(examples) >= limit:
            break
        try:
            examples.append(json.loads(obj_str))
        except json.JSONDecodeError as e:
            print(f"[WARN] Skipping malformed object near char {e.pos}: {e.msg}")

    if examples:
        print(f"[INFO] Loaded {len(examples):,} examples (object-stream format)")
        return examples

    # ── Fallback: JSONL (one object per line) ─────────────────────────────
    examples = []
    for i, line in enumerate(raw.splitlines()):
        if limit and len(examples) >= limit:
            break
        line = line.strip()
        if line:
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    print(f"[INFO] Loaded {len(examples):,} examples (JSONL format)")
    return examples


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(examples: list[dict], show_redacted: bool = True) -> dict:
    """
    Run the rule-based tagger on all examples and evaluate.

    Returns the evaluation metrics dict.
    """
    results       = [process_example(ex) for ex in examples]

    # Collect sequences for evaluation (only where gold labels exist)
    gold_seqs = [r["gold_ner"]      for r in results if r["gold_ner"] is not None]
    pred_seqs = [r["predicted_ner"] for r in results if r["gold_ner"] is not None]

    # ── Show a few redacted sentences ──────────────────────────────────────
    if show_redacted:
        print("\n" + "="*60)
        print("SAMPLE REDACTED OUTPUT")
        print("="*60)
        for r in results[:3]:
            original = " ".join(r["words"])
            print(f"\n  Original : {original}")
            print(f"  Redacted : {r['redacted_text']}")
            print(f"  Gold NER : {r['gold_ner']}")
            print(f"  Pred NER : {r['predicted_ner']}")

    # ── Evaluate ───────────────────────────────────────────────────────────
    metrics = {}
    if gold_seqs:
        metrics = evaluate(gold_seqs, pred_seqs)

        print("\nPer-class Recall (key for PII redaction):")
        pc_recall = per_class_recall(gold_seqs, pred_seqs)
        for entity_type, recall in sorted(pc_recall.items()):
            flag = " ← LOW, needs attention" if recall < 0.6 else ""
            print(f"  {entity_type:<15} {recall:.4f}{flag}")

        confusion_analysis(gold_seqs, pred_seqs, n_examples=5)
    else:
        print("\n[!] No gold labels found — skipping evaluation.")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1: Rule-based Hindi PII NER"
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help=(
            "Path to dataset file (.json / .jsonl). "
            "Auto-detects ../datasets/hi_IndicNER_v1.0/ if omitted."
        )
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max number of examples to load (useful for quick testing)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.data:
        # Explicit path provided on command line
        examples = load_dataset(args.data, limit=args.limit)

    else:
        # Try auto-detecting the dataset one directory up
        detected = _find_default_dataset()
        if detected:
            print(f"[INFO] Auto-detected dataset: {detected}")
            examples = load_dataset(detected, limit=args.limit)
        else:
            print(
                f"[WARN] Dataset not found at {_DEFAULT_DIR}\n"
                f"       Falling back to built-in sample data.\n"
                f"       Run with --data <path> to load your own file."
            )
            examples = SAMPLE_DATA

    run_pipeline(examples, show_redacted=True)
