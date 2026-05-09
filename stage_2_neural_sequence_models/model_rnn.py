"""
model_rnn.py
------------
Model 1: Vanilla RNN for Hindi NER.

PURPOSE IN THE NARRATIVE:
  This model is intentionally the weakest. We build it to demonstrate
  the vanishing gradient problem — not because it's a good NER model.

  Key problem to show your professor:
    - RNN hidden state at step t = tanh(W_h * h_{t-1} + W_x * x_t)
    - Gradient flows back through time via chain rule:
      ∂L/∂h_0 = ∂L/∂h_T * ∏_{t=1}^{T} ∂h_t/∂h_{t-1}
    - Each term is W_h^T * diag(tanh'(·))
    - tanh'(x) ≤ 1 always, ||W_h|| often < 1 after training
    - Product of T such terms → exponentially small for long sequences
    - Hindi sentences: average 15-20 tokens → 15-20 multiplications
    - Entity spans depend on context from far back (e.g. "श्री ... नाम")
    - RNN forgets early context exactly where NER needs it most

  What to observe in the training output:
    - embed_norm (embedding layer gradient) collapses toward 0
    - val_recall plateaus early and stays low
    - The model learns common O tags well but misses long-span entities
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNTagger(nn.Module):
    """
    Vanilla RNN sequence tagger.

    Architecture:
      Embedding → RNN → Linear → (softmax at inference)

    The tanh activation in the RNN cell is what causes vanishing gradients.
    Note: no gating, no separate memory cell, no bidirectionality.
    """

    def __init__(
        self,
        vocab_size:      int,
        num_labels:      int,
        embedding_dim:   int   = 300,
        hidden_dim:      int   = 256,
        num_layers:      int   = 1,
        dropout:         float = 0.3,
        pretrained_emb   = None,   # numpy array (vocab_size, embedding_dim)
        freeze_emb:      bool  = False,
    ):
        super().__init__()

        # ── Embedding layer ────────────────────────────────────────────────
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(
                torch.tensor(pretrained_emb, dtype=torch.float)
            )
        if freeze_emb:
            self.embedding.weight.requires_grad = False

        # ── Vanilla RNN ───────────────────────────────────────────────────
        # nonlinearity='tanh' is the default and is the cause of
        # vanishing gradients. Try nonlinearity='relu' and observe
        # that it helps slightly but causes exploding gradients instead.
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            nonlinearity="tanh",   # ← the problematic activation
        )

        self.dropout  = nn.Dropout(dropout)

        # ── Output projection ─────────────────────────────────────────────
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(
        self,
        token_ids: torch.Tensor,   # (batch, seq_len)
        lengths:   torch.Tensor,   # (batch,) — actual lengths before padding
    ) -> torch.Tensor:             # (batch, seq_len, num_labels)

        # Embed tokens
        emb = self.dropout(self.embedding(token_ids))   # (B, T, E)

        # Pack → RNN → unpack (correctly handles padding)
        packed = pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        rnn_out, _ = self.rnn(packed)
        output, _  = pad_packed_sequence(rnn_out, batch_first=True)  # (B, T, H)

        output = self.dropout(output)

        # Project to label space
        logits = self.classifier(output)   # (B, T, num_labels)
        return logits


# ---------------------------------------------------------------------------
# Factory function — used by run_stage2.py
# ---------------------------------------------------------------------------

def build_rnn(vocab_size, num_labels, embedding_matrix=None, **kwargs):
    return RNNTagger(
        vocab_size=vocab_size,
        num_labels=num_labels,
        pretrained_emb=embedding_matrix,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Expected problems summary (printed by compare_models.py)
# ---------------------------------------------------------------------------

KNOWN_PROBLEMS = """
Vanilla RNN — Known Problems
─────────────────────────────
✗ Vanishing gradients: gradient signal decays exponentially over time steps.
  Embedding layer receives near-zero gradients → slow / no learning of
  long-range context (critical for NER where entity type depends on
  context many tokens away).

✗ No gating: cannot selectively remember or forget information.
  Every hidden state is a full overwrite — early context is lost.

✗ Unidirectional: only sees left context. For Hindi NER, the word
  after an entity often determines its type (e.g. postpositions like
  "ने", "को" following a person name).

→ Solution: LSTM introduces forget/input/output gates to control
  information flow, solving the vanishing gradient problem.
"""
