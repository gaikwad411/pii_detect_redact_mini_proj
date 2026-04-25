"""
redactor.py
-----------
Rule-based NER tagger + PII redactor for Hindi text.

Decision logic (in priority order):
  1. Regex match          → structured PII (phone, aadhaar, PAN …)
  2. Honorific context    → token after "श्री / डॉ / …" → PER
  3. Gazetteer lookup     → PER / LOC / ORG
  4. Morphological suffix → weak signal, used as tiebreaker
  5. Default              → O (not an entity)

BIO scheme:
  B-<TYPE>  → beginning of an entity span
  I-<TYPE>  → continuation of an entity span
  O         → outside any entity

Redaction:
  Entity tokens are replaced with  [REDACTED-<TYPE>]
  Consecutive tokens of the same entity are collapsed into one tag.
"""

from features import extract_token_features


# ---------------------------------------------------------------------------
# Core tagging logic
# ---------------------------------------------------------------------------

def _decide_tag(feat: dict, prev_tag: str) -> str:
    """
    Map a feature dictionary to a BIO tag string.

    Parameters
    ----------
    feat     : output of extract_token_features()
    prev_tag : BIO tag assigned to the previous token (for I- continuation)

    Returns
    -------
    BIO tag string, e.g. "B-PER", "I-PER", "O"
    """

    # ── Rule 1: Regex-detected structured PII ──────────────────────────────
    if feat["regex_tag"]:
        entity_type = feat["regex_tag"]
        # Check if this continues a previous regex entity of the same type
        if prev_tag in (f"B-{entity_type}", f"I-{entity_type}"):
            return f"I-{entity_type}"
        return f"B-{entity_type}"

    # ── Rule 2: Honorific triggers person entity ───────────────────────────
    # If the previous (or two-before) word is an honorific, this token is PER
    if feat["prev_is_honorific"] or feat["prev2_is_honorific"]:
        if prev_tag in ("B-PER", "I-PER"):
            return "I-PER"
        return "B-PER"

    # ── Rule 3: Gazetteer lookup ───────────────────────────────────────────
    if feat["gazetteer_tag"]:
        entity_type = feat["gazetteer_tag"]
        # Continue the span if same entity type as previous
        if prev_tag in (f"B-{entity_type}", f"I-{entity_type}"):
            return f"I-{entity_type}"
        return f"B-{entity_type}"

    # ── Rule 4: Morphological suffix hints (weak signal only) ─────────────
    # Only fire if not preceded by an O — avoids false positives in isolation
    if feat["has_org_suffix"] and prev_tag not in ("O", ""):
        if prev_tag in ("B-ORG", "I-ORG"):
            return "I-ORG"
        return "B-ORG"

    if feat["has_loc_suffix"] and prev_tag not in ("O", ""):
        if prev_tag in ("B-LOC", "I-LOC"):
            return "I-LOC"
        return "B-LOC"

    # ── Default: not an entity ─────────────────────────────────────────────
    return "O"


def tag_sentence(tokens: list[str]) -> list[str]:
    """
    Assign BIO NER tags to every token in a sentence.

    Parameters
    ----------
    tokens : list of word strings

    Returns
    -------
    List of BIO tag strings, same length as tokens.
    """
    tags     = []
    prev_tag = ""

    for idx in range(len(tokens)):
        feat = extract_token_features(tokens, idx)
        tag  = _decide_tag(feat, prev_tag)
        tags.append(tag)
        prev_tag = tag

    return tags


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

def redact_sentence(tokens: list[str], tags: list[str]) -> str:
    """
    Replace entity spans with [REDACTED-<TYPE>] placeholders.

    Consecutive B-/I- tags of the same type are collapsed into
    a single redaction marker.

    Parameters
    ----------
    tokens : original word list
    tags   : BIO tag list (same length)

    Returns
    -------
    Redacted sentence string.
    """
    redacted_tokens = []
    i = 0

    while i < len(tokens):
        tag = tags[i]

        if tag == "O":
            redacted_tokens.append(tokens[i])
            i += 1

        elif tag.startswith("B-"):
            entity_type = tag[2:]          # e.g. "PER", "ORG", "PHONE"

            # Consume all I- tokens that belong to this span
            i += 1
            while i < len(tokens) and tags[i] == f"I-{entity_type}":
                i += 1

            redacted_tokens.append(f"[REDACTED-{entity_type}]")

        else:
            # Orphan I- tag (no preceding B-): treat as entity start
            entity_type = tag[2:]
            redacted_tokens.append(f"[REDACTED-{entity_type}]")
            i += 1

    return " ".join(redacted_tokens)


# ---------------------------------------------------------------------------
# Public API: process one example dict
# ---------------------------------------------------------------------------

def process_example(example: dict) -> dict:
    """
    Process one dataset example (words + optional gold ner).

    Parameters
    ----------
    example : {"words": [...], "ner": [...]}   (ner is optional at inference)

    Returns
    -------
    dict with:
        words        : original tokens
        gold_ner     : gold labels (if present, else None)
        predicted_ner: rule-based predicted labels
        redacted_text: sentence with PII replaced
    """
    tokens    = example["words"]
    gold_ner  = example.get("ner", None)
    pred_ner  = tag_sentence(tokens)
    redacted  = redact_sentence(tokens, pred_ner)

    return {
        "words":         tokens,
        "gold_ner":      gold_ner,
        "predicted_ner": pred_ner,
        "redacted_text": redacted,
    }
