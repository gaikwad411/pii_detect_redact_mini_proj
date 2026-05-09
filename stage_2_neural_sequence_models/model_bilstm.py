"""
model_bilstm.py
---------------
Model 4: Bidirectional LSTM for Hindi NER.

PURPOSE IN THE NARRATIVE:
  BiLSTM is the first model that uses BOTH left and right context
  for every token's prediction. This is crucial for NER.

  Architecture:
    Forward LSTM:  reads left → right,  produces h→_t
    Backward LSTM: reads right → left,  produces h←_t
    Concatenate:   h_t = [h→_t ; h←_t]   (size = 2 * hidden_dim)

  Why bidirectionality matters for Hindi NER specifically:

    Example 1 — postposition confirms entity:
      "अहमद पटेल ने कहा"
       B-PER  I-PER O   O
      The word "ने" (postposition marking agent) comes AFTER the name.
      A forward LSTM tags "पटेल" before seeing "ने".
      A BiLSTM sees "ने" in the backward pass when tagging "पटेल".
      → Right context confirms person entity.

    Example 2 — title before name:
      "राज्यसभा के सदस्य अहमद पटेल"
       B-ORG    O  O      B-PER I-PER
      "सदस्य" (member) before "अहमद" signals a person is coming.
      BiLSTM's backward pass carries this signal when tagging "सदस्य".

  What to observe vs LSTM:
    - Significant improvement in recall, especially for PER
    - Hidden size effectively doubles (we concatenate both directions)
    - Training is ~2x slower than LSTM due to two passes per batch

  Parameter note:
    We pass hidden_dim to nn.LSTM and set bidirectional=True.
    The output dimension is 2 * hidden_dim.
    The classifier layer takes 2 * hidden_dim as input.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMTagger(nn.Module):
    """
    Bidirectional LSTM sequence tagger.

    Architecture:
      Embedding → BiLSTM → Dropout → Linear → (softmax at inference)
    """

    def __init__(
        self,
        vocab_size:      int,
        num_labels:      int,
        embedding_dim:   int   = 300,
        hidden_dim:      int   = 128,   # NOTE: output = 2*hidden_dim = 256
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

        # ── Bidirectional LSTM ─────────────────────────────────────────────
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,          # ← the key change from LSTMTagger
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # Output dim is 2*hidden_dim because we concatenate both directions
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(
        self,
        token_ids: torch.Tensor,
        lengths:   torch.Tensor,
    ) -> torch.Tensor:

        emb    = self.dropout(self.embedding(token_ids))
        packed = pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        bilstm_out, _ = self.bilstm(packed)
        output, _     = pad_packed_sequence(bilstm_out, batch_first=True)
        # output shape: (B, T, 2*hidden_dim) — both directions concatenated
        output = self.dropout(output)
        logits = self.classifier(output)   # (B, T, num_labels)
        return logits


def build_bilstm(vocab_size, num_labels, embedding_matrix=None, **kwargs):
    return BiLSTMTagger(
        vocab_size=vocab_size,
        num_labels=num_labels,
        pretrained_emb=embedding_matrix,
        **kwargs,
    )


KNOWN_PROBLEMS = """
BiLSTM — Known Problems
────────────────────────
✓ Bidirectional: uses full sentence context for every token's tag.
✓ Significant improvement in recall over unidirectional models.
✓ Still solves vanishing gradients via LSTM gating.

✗ Label independence: the classifier predicts each token's label
  independently from a softmax over all labels.
  This means the model CAN predict invalid BIO sequences like:
    O → I-PER  (I- without preceding B-)
    B-PER → I-ORG  (I- of different type than B-)
  These are structurally impossible in valid BIO tagging but nothing
  prevents the model from outputting them.

✗ No explicit transition model between consecutive labels.
  The model learns label correlations implicitly through the LSTM hidden
  state, but this is weaker than modelling transitions explicitly.

→ Solution: add a CRF layer on top that models valid label transitions
  explicitly and finds the globally optimal tag sequence via Viterbi.
"""
