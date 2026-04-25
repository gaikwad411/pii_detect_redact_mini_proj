"""
predict.py
----------
Interactive CLI for Hindi PII detection and redaction.

Usage
-----
# Single sentence (inline):
    python predict.py "अहमद पटेल ने दिल्ली में 9876543210 पर संपर्क किया"

# Interactive mode (keep typing sentences):
    python predict.py

# Show BIO tags alongside redaction:
    python predict.py --tags "अहमद पटेल दिल्ली गए"

# Pipe input:
    echo "डॉ मनमोहन सिंह मुंबई में हैं" | python predict.py
"""

import sys
import argparse

sys.path.insert(0, __import__("os").path.dirname(__file__))

from redactor import tag_sentence, redact_sentence

# ANSI colors — disabled automatically when output is not a terminal
_USE_COLOR = sys.stdout.isatty()

RESET  = "\033[0m"  if _USE_COLOR else ""
BOLD   = "\033[1m"  if _USE_COLOR else ""
RED    = "\033[91m" if _USE_COLOR else ""
GREEN  = "\033[92m" if _USE_COLOR else ""
YELLOW = "\033[93m" if _USE_COLOR else ""
CYAN   = "\033[96m" if _USE_COLOR else ""
DIM    = "\033[2m"  if _USE_COLOR else ""

# Color per entity type for inline highlighting
ENTITY_COLORS = {
    "PER":         "\033[91m" if _USE_COLOR else "",   # red
    "ORG":         "\033[94m" if _USE_COLOR else "",   # blue
    "LOC":         "\033[92m" if _USE_COLOR else "",   # green
    "PHONE":       "\033[93m" if _USE_COLOR else "",   # yellow
    "EMAIL":       "\033[93m" if _USE_COLOR else "",
    "PAN":         "\033[95m" if _USE_COLOR else "",   # magenta
    "AADHAAR":     "\033[95m" if _USE_COLOR else "",
    "DATE":        "\033[96m" if _USE_COLOR else "",   # cyan
    "PINCODE":     "\033[96m" if _USE_COLOR else "",
    "VEHICLE_REG": "\033[96m" if _USE_COLOR else "",
}

TAG_LABELS = {
    "PER":         "Person",
    "ORG":         "Organisation",
    "LOC":         "Location",
    "PHONE":       "Phone",
    "EMAIL":       "Email",
    "PAN":         "PAN Card",
    "AADHAAR":     "Aadhaar",
    "DATE":        "Date",
    "PINCODE":     "Pincode",
    "VEHICLE_REG": "Vehicle Reg.",
}

# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def tokenize(sentence: str) -> list[str]:
    """
    Whitespace tokenizer that:
    - Separates trailing Hindi/English punctuation (। , ! ? ;) as own tokens
    - Does NOT split inside emails (user@example.com) or numbers (9.5, 01/01/24)
    """
    import re
    # Protect emails from being split — replace @ and . temporarily
    # Step 1: separate sentence-ending punctuation that is NOT inside a token
    # Only split a punctuation char if it is surrounded by spaces or at boundary,
    # OR if it's a Devanagari danda (।) which is always standalone punctuation.
    sentence = re.sub(r"(।)", r" \1", sentence)          # danda always split
    sentence = re.sub(r"(?<!\S)([,!?;:])(?!\S)", r" \1 ", sentence)  # standalone punct
    # Split trailing comma/exclamation glued to a Devanagari word (not digits/email)
    sentence = re.sub(r"([\u0900-\u097F])([,!?;:])", r"\1 \2", sentence)
    return sentence.split()


def process_sentence(sentence: str) -> dict:
    """
    Tokenize, tag, redact one sentence string.
    Returns dict with tokens, tags, entities, redacted text.
    """
    tokens   = tokenize(sentence.strip())
    if not tokens:
        return {"tokens": [], "tags": [], "entities": [], "redacted": ""}

    tags     = tag_sentence(tokens)
    redacted = redact_sentence(tokens, tags)
    entities = extract_entities(tokens, tags)

    return {
        "tokens":   tokens,
        "tags":     tags,
        "entities": entities,
        "redacted": redacted,
    }


def extract_entities(tokens: list[str], tags: list[str]) -> list[dict]:
    """
    Extract entity spans from BIO tags.
    Returns list of {"text": ..., "type": ..., "start": ..., "end": ...}
    """
    entities = []
    i = 0
    while i < len(tokens):
        if tags[i].startswith("B-"):
            entity_type = tags[i][2:]
            start = i
            i += 1
            while i < len(tokens) and tags[i] == f"I-{entity_type}":
                i += 1
            span_text = " ".join(tokens[start:i])
            entities.append({
                "text":  span_text,
                "type":  entity_type,
                "label": TAG_LABELS.get(entity_type, entity_type),
                "start": start,
                "end":   i - 1,
            })
        else:
            i += 1
    return entities


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _color_for(entity_type: str) -> str:
    return ENTITY_COLORS.get(entity_type, CYAN)


def print_result(result: dict, show_tags: bool = False) -> None:
    """Pretty-print the analysis result to stdout."""
    tokens   = result["tokens"]
    tags     = result["tags"]
    entities = result["entities"]
    redacted = result["redacted"]

    # ── Highlighted original ──────────────────────────────────────────────
    highlighted = []
    i = 0
    while i < len(tokens):
        tag = tags[i]
        if tag.startswith("B-"):
            entity_type = tag[2:]
            color = _color_for(entity_type)
            span = [tokens[i]]
            i += 1
            while i < len(tokens) and tags[i] == f"I-{entity_type}":
                span.append(tokens[i])
                i += 1
            highlighted.append(f"{color}{BOLD}{' '.join(span)}{RESET}")
        else:
            highlighted.append(tokens[i])
            i += 1

    print(f"\n{DIM}Original :{RESET}  {' '.join(highlighted)}")
    print(f"{DIM}Redacted :{RESET}  {BOLD}{redacted}{RESET}")

    # ── Entity summary ────────────────────────────────────────────────────
    if entities:
        print(f"\n{DIM}Entities detected:{RESET}")
        for ent in entities:
            color = _color_for(ent["type"])
            label = f"{color}{ent['label']:<16}{RESET}"
            print(f"  {label}  {BOLD}{ent['text']}{RESET}")
    else:
        print(f"\n{DIM}No PII entities detected.{RESET}")

    # ── BIO tag table ─────────────────────────────────────────────────────
    if show_tags and tokens:
        print(f"\n{DIM}BIO tags:{RESET}")
        # Column widths
        w = max(len(t) for t in tokens + tags) + 2
        header = "  " + "".join(t.ljust(w) for t in tokens)
        row    = "  " + "".join(
            (f"{_color_for(tag[2:])}{tag}{RESET}" if tag != "O" else DIM + tag + RESET).ljust(
                w + (len(_color_for(tag[2:])) + len(RESET) if tag != "O"
                     else len(DIM) + len(RESET))
            )
            for tag in tags
        )
        print(header)
        print(row)

    print()


def print_legend() -> None:
    print(f"\n{DIM}Entity colour legend:{RESET}")
    for etype, label in TAG_LABELS.items():
        color = _color_for(etype)
        print(f"  {color}■{RESET}  {label}")
    print()


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

BANNER = f"""
{BOLD}{CYAN}Hindi PII Detector — Stage 1 (Rule-based){RESET}
{DIM}Type a Hindi sentence and press Enter.
Commands:  :tags   toggle BIO tag display
           :legend show colour legend
           :help   show this message
           :quit   exit  (or Ctrl-C){RESET}
"""

def run_interactive(show_tags: bool = False) -> None:
    print(BANNER)
    print_legend()

    while True:
        try:
            raw = input(f"{BOLD}>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Bye.{RESET}")
            break

        if not raw:
            continue

        # ── Commands ──────────────────────────────────────────────────────
        if raw == ":quit":
            print(f"{DIM}Bye.{RESET}")
            break
        elif raw == ":tags":
            show_tags = not show_tags
            state = "ON" if show_tags else "OFF"
            print(f"{DIM}BIO tag display: {state}{RESET}")
            continue
        elif raw == ":legend":
            print_legend()
            continue
        elif raw == ":help":
            print(BANNER)
            continue

        result = process_sentence(raw)
        print_result(result, show_tags=show_tags)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hindi PII detector and redactor (rule-based, Stage 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py "अहमद पटेल ने 9876543210 पर फोन किया"
  python predict.py --tags "डॉ मनमोहन सिंह दिल्ली में हैं"
  python predict.py                        # interactive mode
  echo "राहुल गांधी मुंबई गए" | python predict.py
        """
    )
    parser.add_argument(
        "sentence", nargs="?", default=None,
        help="Sentence to analyse. Omit for interactive mode."
    )
    parser.add_argument(
        "--tags", action="store_true",
        help="Show BIO tag table alongside redaction."
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON (useful for piping to other tools)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── Piped input  (echo "..." | python predict.py) ─────────────────────
    if not sys.stdin.isatty() and args.sentence is None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            result = process_sentence(line)
            if args.json:
                import json
                print(json.dumps({
                    "input":    line,
                    "redacted": result["redacted"],
                    "entities": result["entities"],
                    "tags":     result["tags"],
                }, ensure_ascii=False))
            else:
                print(f"\n{DIM}Input:{RESET} {line}")
                print_result(result, show_tags=args.tags)
        sys.exit(0)

    # ── Single sentence from CLI arg ──────────────────────────────────────
    if args.sentence:
        result = process_sentence(args.sentence)
        if args.json:
            import json
            print(json.dumps({
                "input":    args.sentence,
                "redacted": result["redacted"],
                "entities": result["entities"],
                "tags":     result["tags"],
            }, ensure_ascii=False))
        else:
            print_result(result, show_tags=args.tags)
        sys.exit(0)

    # ── Interactive REPL ──────────────────────────────────────────────────
    run_interactive(show_tags=args.tags)
