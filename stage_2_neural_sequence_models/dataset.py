"""
dataset.py
----------
Vocabulary builder, BIO label encoder, PyTorch Dataset and DataLoader.

Shared by all 5 neural models in Stage 2.

Design decisions worth explaining to your professor:
  - <PAD> token at index 0 — allows batch padding without affecting loss
  - <UNK> token at index 1 — handles OOV words at inference time
  - Label padding uses index -1 — PyTorch CrossEntropyLoss ignores it
  - We keep the full BIO scheme (B-/I-/O) — not collapsing to IO
    because CRF in model 5 needs valid transition constraints
"""

import json
import sys
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Special tokens
PAD_TOKEN = "<PAD>"   # index 0  — padding, ignored in loss
UNK_TOKEN = "<UNK>"   # index 1  — out-of-vocabulary words

PAD_LABEL_IDX = -1    # PyTorch CrossEntropyLoss ignores -1 by default


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocabulary:
    """
    Maps tokens ↔ integer indices.
    Built from training data, applied to val/test.
    """

    def __init__(self):
        self.token2idx: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx2token: list[str]      = [PAD_TOKEN, UNK_TOKEN]

    def build(self, sentences: list[list[str]], min_freq: int = 1) -> None:
        """
        Build vocabulary from a list of tokenised sentences.

        Parameters
        ----------
        sentences : list of token lists
        min_freq  : minimum frequency for a token to be included
                    (set > 1 to reduce vocab size on large corpora)
        """
        counts = Counter(tok for sent in sentences for tok in sent)
        for token, freq in sorted(counts.items()):
            if freq >= min_freq and token not in self.token2idx:
                self.token2idx[token] = len(self.idx2token)
                self.idx2token.append(token)

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.token2idx[UNK_TOKEN]
        return [self.token2idx.get(tok, unk) for tok in tokens]

    def __len__(self) -> int:
        return len(self.idx2token)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token2idx, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        v = cls()
        with open(path, encoding="utf-8") as f:
            v.token2idx = json.load(f)
        v.idx2token = [""] * len(v.token2idx)
        for tok, idx in v.token2idx.items():
            v.idx2token[idx] = tok
        return v


# ---------------------------------------------------------------------------
# Label encoder
# ---------------------------------------------------------------------------

class LabelEncoder:
    """
    Maps BIO label strings ↔ integer indices.
    Built once from all labels seen in training data.
    """

    def __init__(self):
        self.label2idx: dict[str, int] = {}
        self.idx2label: list[str]      = []

    def build(self, label_sequences: list[list[str]]) -> None:
        seen = sorted({lbl for seq in label_sequences for lbl in seq})
        for lbl in seen:
            if lbl not in self.label2idx:
                self.label2idx[lbl] = len(self.idx2label)
                self.idx2label.append(lbl)

    def encode(self, labels: list[str]) -> list[int]:
        return [self.label2idx[lbl] for lbl in labels]

    def decode(self, indices: list[int]) -> list[str]:
        return [self.idx2label[i] for i in indices]

    def __len__(self) -> int:
        return len(self.idx2label)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.label2idx, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LabelEncoder":
        le = cls()
        with open(path) as f:
            le.label2idx = json.load(f)
        le.idx2label = [""] * len(le.label2idx)
        for lbl, idx in le.label2idx.items():
            le.idx2label[idx] = lbl
        return le


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class NERDataset(Dataset):
    """
    Converts raw examples into (token_ids, label_ids) tensors.
    """

    def __init__(
        self,
        examples:      list[dict],
        vocab:         Vocabulary,
        label_encoder: LabelEncoder,
    ):
        self.samples = []
        for ex in examples:
            token_ids = vocab.encode(ex["words"])
            label_ids = label_encoder.encode(ex["ner"])
            self.samples.append((
                torch.tensor(token_ids, dtype=torch.long),
                torch.tensor(label_ids, dtype=torch.long),
            ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Collate function — pads variable-length sequences in a batch
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """
    Pads token and label sequences to the length of the longest
    sequence in the batch.

    Token padding: PAD_TOKEN index (0)
    Label padding: PAD_LABEL_IDX (-1) — ignored by CrossEntropyLoss
    """
    token_seqs, label_seqs = zip(*batch)

    # lengths needed for packing in RNN/LSTM/GRU
    lengths = torch.tensor([len(s) for s in token_seqs], dtype=torch.long)

    token_padded = pad_sequence(token_seqs, batch_first=True, padding_value=0)
    label_padded = pad_sequence(
        label_seqs, batch_first=True, padding_value=PAD_LABEL_IDX
    )
    return token_padded, label_padded, lengths


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloader(
    examples:      list[dict],
    vocab:         Vocabulary,
    label_encoder: LabelEncoder,
    batch_size:    int  = 32,
    shuffle:       bool = True,
    num_workers:   int  = 0,
) -> DataLoader:
    dataset = NERDataset(examples, vocab, label_encoder)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )


# ---------------------------------------------------------------------------
# Self-contained dataset loader (no dependency on parent main.py)
# ---------------------------------------------------------------------------

def _iter_json_objects(text: str):
    """
    Yield individual JSON object strings from a file that contains
    multiple objects separated by whitespace only — no commas, no array.
    Uses brace-depth counting to find object boundaries.
    """
    import json as _json
    depth  = 0
    start  = None
    in_str = False
    escape = False

    for i, ch in enumerate(text):
        if escape:          escape = False; continue
        if ch == "\\" and in_str: escape = True; continue
        if ch == '"':       in_str = not in_str; continue
        if in_str:          continue
        if ch == "{":
            if depth == 0:  start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start : i + 1]
                start = None


def _load_dataset(path: str, limit: int | None = None) -> list[dict]:
    """
    Load NER examples from a JSON file.
    Handles three formats automatically:
      1. JSON array  [ {...}, {...} ]
      2. Object-stream  {...}\n{...}\n  (your IndicNER format)
      3. JSONL  one JSON object per line
    """
    import json as _json
    path = Path(path)
    # Resolve relative paths from the script's directory, not cwd
    if not path.is_absolute():
        path = (Path(__file__).parent.parent / path).resolve()
    print(f"[INFO] Loading dataset from: {path}")
    raw = path.read_text(encoding="utf-8")

    # Try JSON array
    stripped = raw.strip()
    if stripped.startswith("["):
        try:
            data = _json.loads(stripped)
            data = data[:limit] if limit else data
            print(f"[INFO] Loaded {len(data):,} examples (JSON array)")
            return data
        except _json.JSONDecodeError:
            pass

    # Object-stream (IndicNER format)
    examples = []
    for obj_str in _iter_json_objects(raw):
        if limit and len(examples) >= limit:
            break
        try:
            examples.append(_json.loads(obj_str))
        except _json.JSONDecodeError:
            pass
    if examples:
        print(f"[INFO] Loaded {len(examples):,} examples (object-stream)")
        return examples

    # JSONL fallback
    for line in raw.splitlines():
        if limit and len(examples) >= limit:
            break
        line = line.strip()
        if line:
            try:
                examples.append(_json.loads(line))
            except _json.JSONDecodeError:
                pass
    print(f"[INFO] Loaded {len(examples):,} examples (JSONL)")
    return examples


# ---------------------------------------------------------------------------
# Data loading helper (re-uses main.py parser logic)
# ---------------------------------------------------------------------------

def load_and_split(
    data_path: str,
    train_ratio: float = 0.8,
    val_ratio:   float = 0.1,
    limit:       int | None = None,
    seed:        int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Load dataset and split into train / val / test.

    Returns (train_examples, val_examples, test_examples)
    """
    import random

    examples = _load_dataset(data_path, limit=limit)

    random.seed(seed)
    random.shuffle(examples)

    n       = len(examples)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train = examples[:n_train]
    val   = examples[n_train : n_train + n_val]
    test  = examples[n_train + n_val :]

    print(f"Split → train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")
    return train, val, test


def build_vocab_and_labels(
    train_examples: list[dict],
    min_freq: int = 1,
) -> tuple[Vocabulary, LabelEncoder]:
    """Build vocabulary and label encoder from training data only."""
    vocab = Vocabulary()
    vocab.build([ex["words"] for ex in train_examples], min_freq=min_freq)

    label_encoder = LabelEncoder()
    label_encoder.build([ex["ner"] for ex in train_examples])

    print(f"Vocabulary size : {len(vocab):,} tokens")
    print(f"Label set       : {label_encoder.idx2label}")
    return vocab, label_encoder
