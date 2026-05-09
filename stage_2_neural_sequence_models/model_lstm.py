"""
model_lstm.py
-------------
Model 2: LSTM for Hindi NER.

PURPOSE IN THE NARRATIVE:
  LSTM (Long Short-Term Memory, Hochreiter & Schmidhuber 1997) was
  designed specifically to solve the vanishing gradient problem.

  The key innovation — three gates + a cell state:

    Forget gate:  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
                  → decides what to ERASE from cell state

    Input gate:   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
    Candidate:    g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)
                  → decides what NEW information to WRITE

    Cell update:  c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
                  → additive update (not multiplicative!) → gradients
                    flow back through addition, not tanh → no vanishing

    Output gate:  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
    Hidden:       h_t = o_t ⊙ tanh(c_t)

  The critical insight for your professor:
    The cell state c_t is updated additively (c_{t-1} + new_info).
    Gradients flowing back through addition do NOT get multiplied by
    a weight matrix at every step — they flow back "for free".
    This is why LSTM can learn dependencies across 50+ tokens.

  What to observe vs RNN:
    - embed_norm stays healthy (doesn't collapse to zero)
    - val_recall improves meaningfully, especially for PER spans
    - Convergence is slower per epoch but reaches a much better optimum
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMTagger(nn.Module):
    """
    Unidirectional LSTM sequence tagger.

    Architecture:
      Embedding → LSTM → Dropout → Linear → (softmax at inference)

    Still unidirectional — we'll fix that in model_bilstm.py.
    """

    def __init__(
        self,
        vocab_size:      int,
        num_labels:      int,
        embedding_dim:   int   = 300,
        hidden_dim:      int   = 256,
        num_layers:      int   = 1,
        dropout:         float = 0.3,
        pretrained_emb         = None,
        freeze_emb:      bool  = False,
    ):
        super().__init__()

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

        # ── LSTM ──────────────────────────────────────────────────────────
        # Notice: same interface as nn.RNN but internally uses 4 gates.
        # Parameter count = 4x that of RNN (one weight matrix per gate).
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(
        self,
        token_ids: torch.Tensor,
        lengths:   torch.Tensor,
    ) -> torch.Tensor:

        emb    = self.dropout(self.embedding(token_ids))
        packed = pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        output, _   = pad_packed_sequence(lstm_out, batch_first=True)
        output      = self.dropout(output)
        logits      = self.classifier(output)
        return logits


def build_lstm(vocab_size, num_labels, embedding_matrix=None, **kwargs):
    return LSTMTagger(
        vocab_size=vocab_size,
        num_labels=num_labels,
        pretrained_emb=embedding_matrix,
        **kwargs,
    )


KNOWN_PROBLEMS = """
LSTM — Known Problems
──────────────────────
✓ Solves vanishing gradients via additive cell state updates.
✓ Gating allows selective memory — learns long-range dependencies.

✗ Still unidirectional: predicts tag for token t using only tokens
  0..t. For NER this is limiting because the word AFTER an entity
  often confirms its type.
  Example: "अहमद पटेल ने" — "ने" (postposition after person) is a
  strong signal that "अहमद पटेल" is a person entity. An LSTM
  processing left-to-right assigns the tag BEFORE seeing "ने".

✗ Parameter-heavy: 4 weight matrices per layer vs 1 for RNN.
  Slower to train, more memory required.

→ Solution 1: GRU — fewer parameters with comparable performance.
→ Solution 2: BiLSTM — reads sequence in both directions.
"""
