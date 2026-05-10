"""
embeddings.py
-------------
Loads pretrained FastText English word vectors and builds an embedding
matrix aligned with our Vocabulary.

TWO LOADERS:
  1. FastTextBinLoader  — uses the native `fasttext` package (.bin format)
       - Supports OOV subword vectors: every token gets a vector, even
         unseen words, via character n-gram composition
       - Requires `fasttext` pip package (compiles from C++ — works on Linux,
         fails on Windows/Python 3.13 due to missing ssize_t / C++17 issues)
       - Use this on GCP Debian VMs

  2. FastTextVecLoader  — uses `gensim` (.vec format)
       - No OOV support: unseen tokens get zero vectors
       - Works on all platforms including Windows + Python 3.13
       - Use this for local Windows development

FILES TO DOWNLOAD (Hindi):
  .bin  (OOV support, ~700MB compressed → ~1.8GB): cc.hi.300.bin
  .vec  (no OOV,     ~2.2GB uncompressed): cc.hi.300.vec.gz → gunzip

  Source: https://fasttext.cc/docs/en/crawl-vectors.html  (Hindi / hi row)

  For GCP: upload once to your bucket, then VMs download from there:
    gsutil cp cc.hi.300.bin gs://<bucket>/embeddings/cc.hi.300.bin

WHY PRETRAINED EMBEDDINGS MATTER:
  - Random init: each token starts as noise, learns only from NER labels
  - FastText .bin: each token starts with 300-dim semantic meaning trained
    on Hindi text — the model only needs to learn how to use that meaning
    for NER, not build it from scratch
  - OOV benefit: Hindi has rich morphology — unseen inflected forms are
    composed from subword n-grams, so coverage stays near 100%
"""

import numpy as np
from pathlib import Path

FASTTEXT_DIM = 300


# ---------------------------------------------------------------------------
# Loader 1: native fasttext package (.bin) — Linux / GCP
# ---------------------------------------------------------------------------

class FastTextBinLoader:
    """
    Loads FastText .bin model using the native `fasttext` package.

    Advantages over .vec:
      - get_word_vector() uses character n-gram subwords for OOV tokens
      - Every token in your vocab gets a non-zero vector
      - No coverage gaps

    Requires: pip install fasttext  (needs build-essential on Debian)
    """

    def __init__(self, bin_path: str):
        self.bin_path = Path(bin_path)
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import fasttext
        except ImportError:
            raise ImportError(
                "fasttext package not found. On Debian: "
                "apt-get install build-essential && pip install fasttext"
            )
        print(f"[INFO] Loading FastText model from {self.bin_path}")
        # suppress fasttext's own progress output
        import fasttext.FastText as _ft
        _ft.eprint = lambda *a, **k: None
        self._model = fasttext.load_model(str(self.bin_path))
        dim = self._model.get_dimension()
        print(f"[INFO] Loaded FastText model  dim={dim}")

    def build_matrix(self, vocab) -> np.ndarray:
        """
        Build embedding matrix of shape (vocab_size, 300).

        Row 0 (<PAD>) -> all zeros
        Row 1 (<UNK>) -> mean of all token vectors
        Row i         -> fasttext.get_word_vector(token)  — always non-zero
        """
        self._load()
        dim        = self._model.get_dimension()
        vocab_size = len(vocab)
        matrix     = np.zeros((vocab_size, dim), dtype=np.float32)

        for idx, token in enumerate(vocab.idx2token):
            if idx == 0:    # <PAD> — keep zero
                continue
            if idx == 1:    # <UNK> — fill after loop
                continue
            matrix[idx] = self._model.get_word_vector(token)

        # <UNK> = mean of all token vectors
        matrix[1] = matrix[2:].mean(axis=0)

        n_tokens = vocab_size - 2
        print(f"[INFO] FastText (bin): {n_tokens:,}/{n_tokens:,} tokens covered "
              f"(100% via subword OOV)")
        return matrix


# ---------------------------------------------------------------------------
# Loader 2: gensim KeyedVectors (.vec) — cross-platform fallback
# ---------------------------------------------------------------------------

class FastTextVecLoader:
    """
    Loads FastText .vec file using gensim KeyedVectors.
    Works on Windows + Python 3.13 with no compilation.

    .vec file format (word2vec text format):
      Line 1: <vocab_size> <dim>
      Line 2+: <word> <300 floats separated by spaces>
    """

    def __init__(self, vec_path: str):
        self.vec_path = Path(vec_path)
        self._kv = None

    def _load(self):
        if self._kv is not None:
            return
        from gensim.models import KeyedVectors
        print(f"[INFO] Loading FastText vectors from {self.vec_path}")
        print(f"       (This takes 2-5 minutes for a large .vec file — one time only)")
        self._kv = KeyedVectors.load_word2vec_format(
            str(self.vec_path),
            binary=False,
            encoding="utf-8",
        )
        print(f"[INFO] Loaded {len(self._kv):,} vectors  dim={self._kv.vector_size}")

    def build_matrix(self, vocab) -> np.ndarray:
        """
        Build embedding matrix of shape (vocab_size, 300).

        Row 0 (<PAD>) -> all zeros
        Row 1 (<UNK>) -> mean of all found vectors
        Row i         -> FastText vector if token in .vec, else zero
        """
        self._load()
        vocab_size = len(vocab)
        matrix     = np.zeros((vocab_size, FASTTEXT_DIM), dtype=np.float32)
        found = 0

        for idx, token in enumerate(vocab.idx2token):
            if idx == 0:    # <PAD> — keep zero
                continue
            if idx == 1:    # <UNK> — fill after loop
                continue
            if token in self._kv:
                matrix[idx] = self._kv[token]
                found += 1

        if found > 0:
            matrix[1] = matrix[2:].mean(axis=0)

        coverage = found / max(vocab_size - 2, 1) * 100
        print(f"[INFO] FastText (vec): {found:,}/{vocab_size-2:,} tokens "
              f"({coverage:.1f}%)")
        if coverage < 50:
            print(f"[WARN] Coverage below 50% — check you downloaded the English "
                  f"file (cc.en.300.vec), not another language.")
        return matrix


# ---------------------------------------------------------------------------
# Random fallback
# ---------------------------------------------------------------------------

def random_embedding_matrix(vocab_size: int, dim: int = FASTTEXT_DIM) -> np.ndarray:
    """Xavier-uniform random init — used when no FastText file is provided."""
    std    = np.sqrt(2.0 / (vocab_size + dim))
    matrix = np.random.normal(0, std, (vocab_size, dim)).astype(np.float32)
    matrix[0] = 0   # <PAD> stays zero
    return matrix


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def load_embeddings(
    vocab,
    fasttext_path: str | None = None,
    embedding_dim: int = FASTTEXT_DIM,
) -> np.ndarray:
    """
    Load pretrained embeddings if path is given and file exists.
    Auto-detects format from file extension:
      .bin  -> FastTextBinLoader  (fasttext package, OOV via subwords)
      .vec  -> FastTextVecLoader  (gensim, no OOV)
    Falls back to random Xavier init if no path given or file missing.

    Returns numpy array of shape (vocab_size, 300).
    """
    if fasttext_path:
        p = Path(fasttext_path)
        if not p.exists():
            print(f"[WARN] FastText file not found at: {p} — falling back to random init")
        elif p.suffix == ".bin":
            loader = FastTextBinLoader(str(p))
            return loader.build_matrix(vocab)
        elif p.suffix == ".vec":
            loader = FastTextVecLoader(str(p))
            return loader.build_matrix(vocab)
        else:
            print(f"[WARN] Unknown FastText format '{p.suffix}' — expected .bin or .vec")

    print(
        "\n[WARNING] No FastText file provided — using random embeddings.\n"
        "  For GCP runs, upload once and the startup script handles the rest:\n"
        "    gsutil cp cc.hi.300.bin gs://<bucket>/embeddings/cc.hi.300.bin\n"
        "  For local Windows runs, download cc.hi.300.vec.gz, extract, and pass:\n"
        "    --fasttext path/to/cc.hi.300.vec\n"
    )
    return random_embedding_matrix(len(vocab), dim=embedding_dim)
