"""
model_gru.py
------------
Model 3: GRU (Gated Recurrent Unit) for Hindi NER.

PURPOSE IN THE NARRATIVE:
  GRU (Cho et al. 2014) is a streamlined version of LSTM.
  It merges the forget and input gates into one "update gate"
  and eliminates the separate cell state — fewer parameters,
  faster training, roughly equal performance.

  GRU equations:
    Reset gate:   r_t = σ(W_r · [h_{t-1}, x_t])
                  → how much of previous hidden state to FORGET

    Update gate:  z_t = σ(W_z · [h_{t-1}, x_t])
                  → how much to update hidden state (vs keep old)

    Candidate:    h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])
    New hidden:   h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

  Comparison with LSTM:
    LSTM: 4 gate matrices, separate cell state c_t and hidden h_t
    GRU:  2 gate matrices, single hidden state h_t
    → GRU has ~25% fewer parameters than LSTM for same hidden_dim

  What to observe vs LSTM:
    - Faster training (fewer parameters, simpler computation)
    - Similar or slightly lower F1 (task-dependent)
    - Gradient health similar to LSTM — additive updates prevent vanishing
    - This is a practical engineering trade-off, not a fundamental advance

  Key point for your professor:
    GRU vs LSTM is an empirical question. For Hindi NER specifically,
    try both and compare. The dataset size and sequence length determine
    which wins. GRU often wins on smaller datasets.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GRUTagger(nn.Module):
    """
    Unidirectional GRU sequence tagger.

    Architecture identical to LSTMTagger except nn.GRU replaces nn.LSTM.
    Keeping the architecture identical isolates the effect of the cell type.
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

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(
                torch.tensor(pretrained_emb, dtype=torch.float)
            )
        if freeze_emb:
            self.embedding.weight.requires_grad = False

        # ── GRU — drop-in replacement for LSTM ────────────────────────────
        self.gru = nn.GRU(
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
        gru_out, _ = self.gru(packed)
        output, _  = pad_packed_sequence(gru_out, batch_first=True)
        output     = self.dropout(output)
        logits     = self.classifier(output)
        return logits


def build_gru(vocab_size, num_labels, embedding_matrix=None, **kwargs):
    return GRUTagger(
        vocab_size=vocab_size,
        num_labels=num_labels,
        pretrained_emb=embedding_matrix,
        **kwargs,
    )


KNOWN_PROBLEMS = """
GRU — Known Problems
─────────────────────
✓ Solves vanishing gradients (same mechanism as LSTM).
✓ Fewer parameters than LSTM → faster training.
✓ Often matches LSTM performance on smaller datasets.

✗ Still unidirectional — same limitation as LSTM.
  Cannot use right-context to inform current token's tag.

✗ Merged gates may lose expressiveness vs LSTM on very long
  sequences or large datasets (empirically task-dependent).

→ The unidirectional limitation is the critical remaining problem.
→ Solution: BiLSTM — process sequence in both directions simultaneously.
"""
