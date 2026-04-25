"""
features.py
-----------
Token-level feature extraction for Hindi NER.

These features are used to understand a token's identity beyond just its
surface form. In Stage 2 (CRF), these become the input feature vector.
Here in Stage 1, they inform rule decisions.

Key insight: Hindi doesn't use capitalisation (unlike English NER), so
we rely on:
  - Script/character properties (Devanagari range: U+0900–U+097F)
  - Morphological suffix/prefix patterns
  - Positional context (what came before this token?)
  - Gazetteer membership
"""

import unicodedata
from rules import (
    PERSON_HONORIFICS,
    get_gazetteer_tag,
    get_regex_tag,
)

# ---------------------------------------------------------------------------
# Script / character utilities
# ---------------------------------------------------------------------------

DEVANAGARI_START = 0x0900
DEVANAGARI_END   = 0x097F

def is_devanagari(token: str) -> bool:
    """True if every non-space character is in the Devanagari Unicode block."""
    return all(
        DEVANAGARI_START <= ord(ch) <= DEVANAGARI_END
        for ch in token
        if not ch.isspace()
    )

def is_numeric(token: str) -> bool:
    """True for ASCII digits or Devanagari digits (०–९)."""
    devanagari_digits = set("०१२३४५६७८९")
    return all(ch.isdigit() or ch in devanagari_digits for ch in token)

def is_mixed_script(token: str) -> bool:
    """True if token contains both Devanagari and Latin characters."""
    has_dev = any(DEVANAGARI_START <= ord(ch) <= DEVANAGARI_END for ch in token)
    has_lat = any(ch.isascii() and ch.isalpha() for ch in token)
    return has_dev and has_lat

def char_type(token: str) -> str:
    """Coarse character-type label for the whole token."""
    if is_numeric(token):
        return "NUM"
    if is_devanagari(token):
        return "DEV"
    if token.isascii():
        return "LAT"          # Latin script (English words, codes)
    if is_mixed_script(token):
        return "MIX"
    return "OTH"

# ---------------------------------------------------------------------------
# Hindi morphological suffix hints
# (These are heuristic — extend based on your corpus analysis)
# ---------------------------------------------------------------------------

# Suffixes that commonly appear at the end of person names in Hindi
PERSON_SUFFIXES = {"जी", "जी।", "साहब"}

# Suffixes that commonly appear at the end of org / place names
ORG_SUFFIXES    = {"कोर्ट", "न्यायालय", "मंत्रालय", "विभाग", "संस्थान",
                   "बोर्ड", "आयोग", "परिषद", "समिति", "सभा", "दल"}

LOC_SUFFIXES    = {"नगर", "पुर", "गढ़", "गाँव", "खेड़ा", "वाड़ा",
                   "पट्टी", "ग्राम", "शहर", "जिला", "तालुका"}

def has_person_suffix(token: str) -> bool:
    return any(token.endswith(s) for s in PERSON_SUFFIXES)

def has_org_suffix(token: str) -> bool:
    return any(token.endswith(s) for s in ORG_SUFFIXES)

def has_loc_suffix(token: str) -> bool:
    return any(token.endswith(s) for s in LOC_SUFFIXES)

# ---------------------------------------------------------------------------
# Main feature extraction
# ---------------------------------------------------------------------------

def extract_token_features(
    tokens: list[str],
    idx: int,
) -> dict:
    """
    Extract a feature dictionary for the token at position `idx`.

    Parameters
    ----------
    tokens : list of words in the sentence
    idx    : index of the current token

    Returns
    -------
    dict of feature_name → value  (used for CRF in Stage 2,
    and for rule logic in Stage 1)
    """
    token = tokens[idx]
    prev_token  = tokens[idx - 1] if idx > 0 else "<START>"
    prev2_token = tokens[idx - 2] if idx > 1 else "<START>"
    next_token  = tokens[idx + 1] if idx < len(tokens) - 1 else "<END>"

    gazetteer_tag = get_gazetteer_tag(token)
    regex_tag     = get_regex_tag(token)

    features = {
        # Surface form
        "token":            token,
        "token_len":        len(token),

        # Script
        "char_type":        char_type(token),
        "is_devanagari":    is_devanagari(token),
        "is_numeric":       is_numeric(token),

        # Morphological hints
        "has_person_suffix": has_person_suffix(token),
        "has_org_suffix":    has_org_suffix(token),
        "has_loc_suffix":    has_loc_suffix(token),

        # Context — previous tokens
        "prev_token":       prev_token,
        "prev2_token":      prev2_token,
        "prev_is_honorific": prev_token in PERSON_HONORIFICS,
        "prev2_is_honorific": prev2_token in PERSON_HONORIFICS,

        # Context — next token
        "next_token":       next_token,

        # Gazetteer & regex
        "gazetteer_tag":    gazetteer_tag,   # "PER" / "LOC" / "ORG" / None
        "regex_tag":        regex_tag,       # "PHONE" / "PAN" / ... / None

        # Position
        "is_first":         idx == 0,
        "is_last":          idx == len(tokens) - 1,
        "position_ratio":   round(idx / max(len(tokens) - 1, 1), 2),
    }

    return features
