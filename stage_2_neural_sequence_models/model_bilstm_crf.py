"""
model_bilstm_crf.py
-------------------
Model 5: BiLSTM + CRF — the gold standard pre-transformer NER model.

PURPOSE IN THE NARRATIVE:
  This model adds a CRF (Conditional Random Field) layer on top of
  BiLSTM. The CRF solves the label independence problem by modelling
  the joint probability of the entire label sequence.

  How the CRF layer works:
    BiLSTM outputs "emission scores" e(y_t | x) for each token t
    and each possible label y_t.

    The CRF adds "transition scores" T[y_{t-1}, y_t] — a learnable
    matrix where T[i, j] = score of transitioning from label i to label j.

    Inference (Viterbi algorithm):
      Find the label sequence y* that maximises:
        score(y, x) = Σ_t e(y_t | x_t) + Σ_t T[y_{t-1}, y_t]
      This is solved exactly in O(T * K²) by dynamic programming.

    Training (negative log-likelihood):
      Loss = -log P(y* | x) = -score(y*, x) + log Σ_{y} exp(score(y, x))
      The partition function (log-sum-exp) is computed via forward algorithm.

  Why CRF matters for NER specifically:
    Without CRF: P(y_1, ..., y_T | x) = Π_t P(y_t | x)  (independent)
    With CRF:    P(y_1, ..., y_T | x) ∝ exp(Σ emissions + Σ transitions)

    The transition matrix T learns constraints like:
      T[O, I-PER] = very negative   (I- can't follow O)
      T[B-PER, I-PER] = positive    (I- commonly follows B- of same type)
      T[B-PER, I-ORG] = very negative (I- of different type is invalid)

    These structural constraints are exactly what NER requires.

  What to observe vs BiLSTM:
    - Eliminates invalid BIO sequences in predictions
    - Improvement in F1 on entity boundaries (B-/I- transitions)
    - Slightly slower inference (Viterbi vs argmax), but negligible
    - Best overall recall and F1 among all 5 models

Implementation note:
  We use the pytorch-crf library (torchcrf.CRF).
  The CRF layer handles:
    - Forward algorithm (partition function for loss)
    - Viterbi decoding (best sequence for inference)
    - Mask handling (ignores PAD positions)
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF


class BiLSTMCRFTagger(nn.Module):
    """
    BiLSTM + CRF sequence tagger.

    The BiLSTM produces emission scores.
    The CRF layer finds the globally optimal label sequence.
    """

    def __init__(
        self,
        vocab_size:      int,
        num_labels:      int,
        embedding_dim:   int   = 300,
        hidden_dim:      int   = 128,
        num_layers:      int   = 1,
        dropout:         float = 0.3,
        pretrained_emb         = None,
        freeze_emb:      bool  = False,
    ):
        super().__init__()

        self.num_labels = num_labels

        # ── Embedding ─────────────────────────────────────────────────────
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(
                torch.tensor(pretrained_emb, dtype=torch.float)
            )
        if freeze_emb:
            self.embedding.weight.requires_grad = False

        # ── BiLSTM ────────────────────────────────────────────────────────
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # ── Emission scorer (BiLSTM → label scores) ───────────────────────
        self.hidden2emissions = nn.Linear(hidden_dim * 2, num_labels)

        # ── CRF layer ─────────────────────────────────────────────────────
        # batch_first=True to match our (B, T, C) tensor convention
        self.crf = CRF(num_labels, batch_first=True)

    def _get_emissions(
        self,
        token_ids: torch.Tensor,
        lengths:   torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run embedding + BiLSTM to get emission scores and padding mask.

        Returns:
            emissions : (B, T, num_labels)
            mask      : (B, T) bool tensor — True for real tokens
        """
        emb    = self.dropout(self.embedding(token_ids))
        packed = pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        bilstm_out, _ = self.bilstm(packed)
        output, _     = pad_packed_sequence(bilstm_out, batch_first=True)
        output        = self.dropout(output)
        emissions     = self.hidden2emissions(output)   # (B, T, num_labels)

        # Build mask: True for positions that are real tokens (not PAD)
        B, T = token_ids.shape
        mask = torch.zeros(B, T, dtype=torch.bool, device=token_ids.device)
        for i, L in enumerate(lengths):
            mask[i, :L] = True

        return emissions, mask

    def forward(
        self,
        token_ids: torch.Tensor,
        lengths:   torch.Tensor,
    ) -> list[list[int]]:
        """
        Inference: returns Viterbi-decoded best label sequences.

        Returns list of lists of label indices (variable length per sequence).
        The trainer's predict_sequences() handles this via use_crf=True.
        """
        emissions, mask = self._get_emissions(token_ids, lengths)
        # CRF.decode() returns the best path (Viterbi)
        return self.crf.decode(emissions, mask=mask)

    def loss(
        self,
        token_ids: torch.Tensor,
        label_ids: torch.Tensor,
        lengths:   torch.Tensor,
    ) -> torch.Tensor:
        """
        Training: returns negative log-likelihood loss.

        The CRF loss = -log P(y* | x)
                     = -(gold sequence score) + log(partition function)
        Computed via the forward algorithm (dynamic programming).

        Note: label_ids may contain -1 (PAD) — we replace with 0 before
        passing to CRF (the mask ensures PAD positions are ignored).
        """
        emissions, mask = self._get_emissions(token_ids, lengths)
        # CRF requires non-negative label indices even in masked positions
        labels_safe = label_ids.clone()
        labels_safe[labels_safe == -1] = 0
        # CRF.forward() returns log-likelihood; negate for loss
        log_likelihood = self.crf(emissions, labels_safe, mask=mask)
        return -log_likelihood


def build_bilstm_crf(vocab_size, num_labels, embedding_matrix=None, **kwargs):
    return BiLSTMCRFTagger(
        vocab_size=vocab_size,
        num_labels=num_labels,
        pretrained_emb=embedding_matrix,
        **kwargs,
    )


KNOWN_PROBLEMS = """
BiLSTM + CRF — Known Problems
───────────────────────────────
✓ Globally optimal label sequences via Viterbi decoding.
✓ Transition constraints eliminate invalid BIO sequences.
✓ Bidirectional context + explicit label dependencies.
✓ State-of-the-art pre-transformer NER performance.

✗ Fixed context window: BiLSTM hidden state compresses arbitrarily
  long context into a fixed-size vector. For very long documents
  (not sentences), context from far away may still be lost.

✗ No subword modelling: our vocabulary treats "पटेल" and "पटेलों"
  (oblique plural) as completely different tokens. For morphologically
  rich languages like Hindi, this is a real limitation.
  FastText embeddings partially mitigate this in the embedding layer.

✗ No pre-training on large text: the BiLSTM is trained only on our
  labelled NER data. It doesn't benefit from the distributional
  knowledge a language model learns from billions of tokens.

→ Solution: Transformer-based models (MuRIL, mBERT) in Stage 3
  address all three: self-attention handles arbitrary context,
  subword tokenisation handles morphology, and pre-training on
  large corpora provides rich linguistic knowledge.
"""
