"""
trainer.py
----------
Shared training loop used by all 5 neural models.

Key features:
  - Early stopping on validation F1 (saves best checkpoint)
  - Gradient norm logging (lets us *see* vanishing gradients in RNN)
  - Per-epoch seqeval evaluation on val set
  - Clean results dict for compare_models.py

The gradient norm logging is pedagogically important:
  You can literally show your professor a chart of gradient norms
  collapsing toward zero in the RNN vs staying healthy in LSTM/GRU.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from seqeval.metrics import f1_score, recall_score, precision_score
from seqeval.scheme import IOB2


# ---------------------------------------------------------------------------
# Device — CPU training
# ---------------------------------------------------------------------------
# Explicitly CPU. Training time estimates with hidden_dim=128, batch=16,
# ~50k examples (use --limit to control this):
#   RNN        ~10–15 min/epoch  (early stops ~10 epochs) → ~2 hrs total
#   LSTM       ~15–20 min/epoch  (early stops ~12 epochs) → ~3 hrs total
#   GRU        ~12–18 min/epoch  (early stops ~10 epochs) → ~2 hrs total
#   BiLSTM     ~20–25 min/epoch  (early stops ~12 epochs) → ~4 hrs total
#   BiLSTM+CRF ~25–30 min/epoch  (early stops ~12 epochs) → ~5 hrs total
#
# Practical tip: run with --limit 10000 first to sanity-check everything
# (~15 min total for all 5 models), then run --limit 50000 for real results.

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Gradient norm utility
# ---------------------------------------------------------------------------

def compute_grad_norm(model: nn.Module) -> float:
    """
    Compute the L2 norm of all gradients in the model.

    A healthy model: norm stays roughly constant across epochs.
    Vanishing gradients (RNN): norm collapses toward 0 in early layers.
    Exploding gradients:       norm spikes → NaN loss.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def compute_first_layer_grad_norm(model: nn.Module) -> float:
    """
    Compute gradient norm of the embedding layer only.

    This is where vanishing gradients are most visible:
    the embedding layer is furthest from the loss in the
    computation graph, so gradients arrive weakest here.

    In a vanilla RNN on long sequences, this will be near-zero
    after just a few epochs — demonstrating the vanishing problem.
    """
    for name, p in model.named_parameters():
        if "embed" in name.lower() and p.grad is not None:
            return p.grad.data.norm(2).item()
    return 0.0


# ---------------------------------------------------------------------------
# Sequence prediction (shared by non-CRF models)
# ---------------------------------------------------------------------------

def predict_sequences(
    model:         nn.Module,
    dataloader:    DataLoader,
    label_encoder,
    use_crf:       bool = False,
) -> tuple[list[list[str]], list[list[str]]]:
    """
    Run model in eval mode, return (gold_sequences, pred_sequences)
    as BIO label strings — ready for seqeval.

    Handles PAD positions (label = -1) correctly by masking them out.
    """
    model.eval()
    gold_all = []
    pred_all = []

    with torch.no_grad():
        for token_ids, label_ids, lengths in dataloader:
            token_ids = token_ids.to(DEVICE)
            label_ids = label_ids.to(DEVICE)

            if use_crf:
                # CRF model returns decoded label sequences directly
                pred_label_ids = model(token_ids, lengths)
            else:
                logits = model(token_ids, lengths)         # (B, T, num_labels)
                pred_label_ids = logits.argmax(dim=-1)     # (B, T)

            # Convert to string labels, stripping PAD positions
            B = token_ids.size(0)
            for i in range(B):
                L   = lengths[i].item()
                gold = label_encoder.decode(label_ids[i, :L].tolist())
                if use_crf:
                    pred = label_encoder.decode(pred_label_ids[i])
                else:
                    pred = label_encoder.decode(pred_label_ids[i, :L].tolist())
                gold_all.append(gold)
                pred_all.append(pred)

    return gold_all, pred_all


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Shared training loop for all Stage 2 models.

    Usage:
        trainer = Trainer(model, optimizer, label_encoder)
        results = trainer.train(train_loader, val_loader, epochs=20)
    """

    def __init__(
        self,
        model:         nn.Module,
        optimizer:     torch.optim.Optimizer,
        label_encoder,
        model_name:    str  = "model",
        use_crf:       bool = False,
        clip_grad:     float = 5.0,
        patience:      int  = 5,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model          = model.to(DEVICE)
        self.optimizer      = optimizer
        self.label_encoder  = label_encoder
        self.model_name     = model_name
        self.use_crf        = use_crf
        self.clip_grad      = clip_grad    # gradient clipping threshold
        self.patience       = patience     # early stopping patience
        self.checkpoint_dir = checkpoint_dir

        import os
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Loss: ignore PAD positions (index -1)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    # ── Single training epoch ──────────────────────────────────────────────

    def _train_epoch(self, dataloader: DataLoader) -> dict:
        self.model.train()
        total_loss   = 0.0
        total_tokens = 0
        grad_norms   = []
        embed_norms  = []

        for token_ids, label_ids, lengths in dataloader:
            token_ids = token_ids.to(DEVICE)
            label_ids = label_ids.to(DEVICE)

            self.optimizer.zero_grad()

            if self.use_crf:
                # CRF model returns negative log-likelihood directly
                loss = self.model.loss(token_ids, label_ids, lengths)
            else:
                logits = self.model(token_ids, lengths)    # (B, T, C)
                B, T, C = logits.shape
                loss = self.criterion(
                    logits.view(B * T, C),
                    label_ids.view(B * T),
                )

            loss.backward()

            # ── Record gradient norms BEFORE clipping ─────────────────────
            grad_norms.append(compute_grad_norm(self.model))
            embed_norms.append(compute_first_layer_grad_norm(self.model))

            # ── Gradient clipping (prevents exploding gradients) ──────────
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

            self.optimizer.step()

            n_tokens = (label_ids != -1).sum().item()
            total_loss   += loss.item() * n_tokens
            total_tokens += n_tokens

        return {
            "loss":           total_loss / max(total_tokens, 1),
            "grad_norm_mean": sum(grad_norms) / len(grad_norms),
            "grad_norm_min":  min(grad_norms),
            "embed_norm_mean": sum(embed_norms) / len(embed_norms),
        }

    # ── Validation ─────────────────────────────────────────────────────────

    def _validate(self, dataloader: DataLoader) -> dict:
        gold, pred = predict_sequences(
            self.model, dataloader, self.label_encoder, self.use_crf
        )
        f1  = f1_score(gold, pred, average="macro",
                       scheme=IOB2, zero_division=0)
        rec = recall_score(gold, pred, average="macro",
                           scheme=IOB2, zero_division=0)
        pre = precision_score(gold, pred, average="macro",
                              scheme=IOB2, zero_division=0)
        return {"f1": f1, "recall": rec, "precision": pre}

    # ── Full training loop ─────────────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        epochs:       int = 30,
    ) -> dict:
        """
        Train for up to `epochs` epochs with early stopping on val F1.

        Returns a results dict with:
          - per_epoch: list of dicts with loss, grad norms, train/val metrics
          - best_val_f1, best_val_recall, best_epoch
          - total_time_seconds
          - gradient_problem: description of what we observed
        """
        best_val_f1    = 0.0
        best_epoch     = 0
        no_improve     = 0
        per_epoch      = []
        start_time     = time.time()

        checkpoint_path = f"{self.checkpoint_dir}/{self.model_name}_best.pt"

        print(f"\n{'='*72}")
        print(f"Training {self.model_name}  |  device={DEVICE}")
        print(f"{'='*72}")
        print(f"{'Epoch':>6}  {'Loss':>8}  {'GradNorm':>10}  "
              f"{'EmbNorm':>10}  {'TrainF1':>8}  {'Val F1':>8}  {'Gap':>7}")
        print("-" * 72)

        for epoch in range(1, epochs + 1):
            train_metrics = self._train_epoch(train_loader)
            train_eval    = self._validate(train_loader)   # train F1
            val_metrics   = self._validate(val_loader)

            gap = val_metrics["f1"] - train_eval["f1"]    # negative = overfit

            row = {
                "epoch":          epoch,
                "loss":           round(train_metrics["loss"], 4),
                "grad_norm":      round(train_metrics["grad_norm_mean"], 4),
                "embed_norm":     round(train_metrics["embed_norm_mean"], 6),
                "train_f1":       round(train_eval["f1"], 4),
                "train_recall":   round(train_eval["recall"], 4),
                "val_f1":         round(val_metrics["f1"], 4),
                "val_recall":     round(val_metrics["recall"], 4),
                "val_precision":  round(val_metrics["precision"], 4),
                "overfit_gap":    round(gap, 4),           # val_f1 - train_f1
            }
            per_epoch.append(row)

            gap_str = f"{gap:+.4f}"
            print(
                f"{epoch:>6}  {row['loss']:>8.4f}  "
                f"{row['grad_norm']:>10.4f}  {row['embed_norm']:>10.6f}  "
                f"{row['train_f1']:>8.4f}  {row['val_f1']:>8.4f}  {gap_str:>7}"
            )

            # ── Early stopping ────────────────────────────────────────────
            if val_metrics["f1"] > best_val_f1:
                best_val_f1    = val_metrics["f1"]
                best_val_recall = val_metrics["recall"]
                best_epoch     = epoch
                no_improve     = 0
                torch.save(self.model.state_dict(), checkpoint_path)
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"\n[Early stop] No improvement for {self.patience} "
                          f"epochs. Best epoch: {best_epoch}")
                    break

        total_time = time.time() - start_time

        # ── Reload best checkpoint ────────────────────────────────────────
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print(f"\nBest val F1: {best_val_f1:.4f}  "
              f"Recall: {best_val_recall:.4f}  "
              f"Epoch: {best_epoch}  "
              f"Time: {total_time:.1f}s")

        # ── Overfitting summary ───────────────────────────────────────────
        best_row  = per_epoch[best_epoch - 1]
        train_f1  = best_row["train_f1"]
        overfit   = best_row["overfit_gap"]          # val_f1 - train_f1
        print(f"\nOverfit check @ best epoch {best_epoch}:")
        print(f"  Train F1 : {train_f1:.4f}")
        print(f"  Val   F1 : {best_val_f1:.4f}")
        print(f"  Gap      : {overfit:+.4f}  "
              f"({'generalising well' if overfit > -0.05 else 'overfitting — val << train'})")

        # ── Diagnose gradient behaviour ───────────────────────────────────
        gradient_problem = _diagnose_gradients(self.model_name, per_epoch)

        return {
            "model_name":       self.model_name,
            "best_val_f1":      best_val_f1,
            "best_val_recall":  best_val_recall,
            "best_train_f1":    train_f1,
            "overfit_gap":      overfit,
            "best_epoch":       best_epoch,
            "total_time":       round(total_time, 1),
            "per_epoch":        per_epoch,
            "gradient_problem": gradient_problem,
            "checkpoint_path":  checkpoint_path,
        }


# ---------------------------------------------------------------------------
# Gradient problem diagnosis
# ---------------------------------------------------------------------------

def _diagnose_gradients(model_name: str, per_epoch: list[dict]) -> str:
    """
    Analyse the gradient norm history and return a human-readable
    description of what gradient problem (if any) was observed.

    This is the key academic insight from running the RNN:
      - Vanishing: embed_norm collapses from epoch 1 onward
      - Exploding: grad_norm spikes suddenly (usually epoch 1-2)
      - Healthy:   grad_norm stays roughly stable
    """
    if len(per_epoch) < 3:
        return "Insufficient epochs to diagnose."

    embed_norms = [r["embed_norm"] for r in per_epoch]
    grad_norms  = [r["grad_norm"]  for r in per_epoch]

    first_embed = embed_norms[0] if embed_norms[0] > 0 else 1e-9
    last_embed  = embed_norms[-1]
    ratio       = last_embed / first_embed

    max_grad    = max(grad_norms)
    mean_grad   = sum(grad_norms) / len(grad_norms)

    if max_grad > 10 * mean_grad:
        return (
            f"EXPLODING GRADIENTS detected in {model_name}. "
            f"Max grad norm {max_grad:.2f} >> mean {mean_grad:.2f}. "
            f"Gradient clipping at 5.0 was applied but spikes occurred."
        )
    elif ratio < 0.1:
        return (
            f"VANISHING GRADIENTS detected in {model_name}. "
            f"Embedding layer gradient norm dropped from {first_embed:.6f} "
            f"to {last_embed:.6f} ({ratio*100:.1f}% remaining). "
            f"The model struggles to learn long-range dependencies. "
            f"→ This motivates LSTM with gating mechanisms."
        )
    elif ratio < 0.3:
        return (
            f"MILD VANISHING detected in {model_name}. "
            f"Embedding gradient norm reduced to {ratio*100:.1f}% of initial. "
            f"Learning is slow for tokens involved in long dependencies."
        )
    else:
        return (
            f"HEALTHY GRADIENTS in {model_name}. "
            f"Embedding gradient norm ratio: {ratio*100:.1f}%. "
            f"Gating mechanisms successfully prevent vanishing."
        )
