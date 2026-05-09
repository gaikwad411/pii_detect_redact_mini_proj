"""
embeddings.py
-------------
Loads pretrained FastText Hindi word vectors and builds an embedding
matrix aligned with our Vocabulary.

WHY GENSIM INSTEAD OF THE FASTTEXT PACKAGE:
  The official `fasttext` Python package does not provide pre-built
  Windows wheels for Python 3.13. It requires compiling C++ source with
  MSVC, which fails due to missing `ssize_t` and C++17 issues.

  We use `gensim` instead, which:
    - Has pre-built wheels for all platforms including Windows + Python 3.13
    - Loads the FastText .vec format natively via KeyedVectors
    - No compilation required — pure pip install

FILE TO DOWNLOAD:
  Go to: https://fasttext.cc/docs/en/crawl-vectors.html
  Find "Hindi" and download: cc.hi.300.vec.gz   (~700MB compressed)

  Extract it (gives you cc.hi.300.vec, ~2.2GB), then pass:
    --fasttext path/to/cc.hi.300.vec

  Note on .bin vs .vec:
    The .bin format supports OOV subword vectors (better for Hindi morphology)
    but requires the fasttext package which won't build on Windows/Python 3.13.
    The .vec format has no OOV support — unseen tokens get zero vectors —
    but coverage on a 1M sentence Hindi corpus is still very high (~85-90%).

WHY PRETRAINED EMBEDDINGS MATTER (for your professor):
  - Random init: each token starts as noise, learns only from NER labels
  - FastText .vec: each token starts with 300-dim semantic meaning from
    157 billion tokens of Hindi text — the model only needs to learn
    how to use that meaning for NER, not build it from scratch
  - This is the difference between learning "अहमद is a name" vs
    learning "अहमद means something similar to other names it co-occurs with"
"""

import numpy as np
from pathlib import Path

FASTTEXT_DIM = 300


# ---------------------------------------------------------------------------
# Gensim-based loader (.vec format)
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
        print(f"       (This takes 2-5 minutes for a 2GB .vec file — one time only)")
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
            if idx == 0:        # <PAD> — keep zero
                continue
            if idx == 1:        # <UNK> — fill after loop
                continue
            if token in self._kv:
                matrix[idx] = self._kv[token]
                found += 1

        # <UNK> = mean of all found vectors — better than random
        if found > 0:
            matrix[1] = matrix[2:].mean(axis=0)

        coverage = found / max(vocab_size - 2, 1) * 100
        print(f"[INFO] Embedding coverage: {found:,}/{vocab_size-2:,} tokens "
              f"({coverage:.1f}%)")
        if coverage < 50:
            print(f"[WARN] Coverage below 50% — check that you downloaded "
                  f"the Hindi file (cc.hi.300.vec), not another language.")
        return matrix


# ---------------------------------------------------------------------------
# Random fallback
# ---------------------------------------------------------------------------

def random_embedding_matrix(vocab_size: int, dim: int = FASTTEXT_DIM) -> np.ndarray:
    """
    Xavier-uniform random init — used when no .vec file is provided.
    Visibly worse results, but lets you run without downloading 2GB.
    """
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
    Load pretrained embeddings if path is given and file exists,
    otherwise fall back to random init.

    Parameters
    ----------
    vocab         : Vocabulary object
    fasttext_path : path to cc.hi.300.vec  (NOT .bin on Windows)
    embedding_dim : only used for random fallback

    Returns
    -------
    numpy array of shape (vocab_size, 300)
    """
    if fasttext_path:
        p = Path(fasttext_path)
        if not p.exists():
            print(f"[WARN] FastText file not found at: {p}")
        elif p.suffix != ".vec":
            print(
                f"[WARN] On Windows/Python 3.13 only .vec format is supported.\n"
                f"       Download cc.hi.300.vec.gz directly from:\n"
                f"       https://fasttext.cc/docs/en/crawl-vectors.html"
            )
        else:
            loader = FastTextVecLoader(str(p))
            return loader.build_matrix(vocab)

    print(
        "\n[WARNING] No FastText .vec file provided — using random embeddings.\n"
        "  To use pretrained vectors:\n"
        "    1. Download cc.hi.300.vec.gz from:\n"
        "       https://fasttext.cc/docs/en/crawl-vectors.html  (Hindi section)\n"
        "    2. Extract: use 7-zip or 'gunzip cc.hi.300.vec.gz' in WSL\n"
        "    3. Run with: --fasttext path/to/cc.hi.300.vec\n"
        "  Training will work without it but results will be noticeably worse.\n"
    )
    return random_embedding_matrix(len(vocab), dim=embedding_dim)
