"""
run_stage2.py
-------------
Master script: trains all 5 neural models sequentially and prints
a progression table showing improvement and problems at each step.

Usage
-----
# Full run on your dataset:
    python run_stage2.py --data ../../datasets/hi_IndicNER_v1.0/train.json

# Quick smoke-test (1000 examples, 3 epochs):
    python run_stage2.py --data ../../datasets/hi_IndicNER_v1.0/train.json \\
        --limit 1000 --epochs 3

# With pretrained FastText embeddings:
    python run_stage2.py --data ../../datasets/hi_IndicNER_v1.0/train.json \\
        --fasttext path/to/cc.hi.300.bin

# Skip already-trained models (loads checkpoint):
    python run_stage2.py --data ... --skip rnn lstm
"""

import sys
import os
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from dataset         import load_and_split, build_vocab_and_labels, make_dataloader
from embeddings      import load_embeddings
from trainer         import Trainer, DEVICE, predict_sequences
from evaluate_neural import full_eval, print_progression_table, print_gradient_analysis

from model_rnn        import build_rnn,        KNOWN_PROBLEMS as RNN_PROBLEMS
from model_lstm       import build_lstm,       KNOWN_PROBLEMS as LSTM_PROBLEMS
from model_gru        import build_gru,        KNOWN_PROBLEMS as GRU_PROBLEMS
from model_bilstm     import build_bilstm,     KNOWN_PROBLEMS as BILSTM_PROBLEMS
from model_bilstm_crf import build_bilstm_crf, KNOWN_PROBLEMS as BILSTM_CRF_PROBLEMS


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "rnn":         (build_rnn,        RNN_PROBLEMS),
    "lstm":        (build_lstm,       LSTM_PROBLEMS),
    "gru":         (build_gru,        GRU_PROBLEMS),
    "bilstm":      (build_bilstm,     BILSTM_PROBLEMS),
    "bilstm_crf":  (build_bilstm_crf, BILSTM_CRF_PROBLEMS),
}

MODEL_DISPLAY_NAMES = {
    "rnn":        "RNN",
    "lstm":       "LSTM",
    "gru":        "GRU",
    "bilstm":     "BiLSTM",
    "bilstm_crf": "BiLSTM+CRF",
}


# ---------------------------------------------------------------------------
# Train one model
# ---------------------------------------------------------------------------

def train_one_model(
    model_key:      str,
    vocab,
    label_encoder,
    embedding_matrix,
    train_loader,
    val_loader,
    test_loader,
    epochs:         int,
    patience:       int,
    hidden_dim:     int,
    checkpoint_dir: str,
    skip:           list[str],
) -> tuple[dict, dict]:
    """
    Train a single model, evaluate on test set, return (train_result, eval_result).
    If model_key in skip, loads checkpoint instead of training.
    """
    build_fn, known_problems = MODEL_REGISTRY[model_key]
    display_name = MODEL_DISPLAY_NAMES[model_key]
    use_crf = (model_key == "bilstm_crf")

    print(f"\n{'#'*60}")
    print(f"# Model: {display_name}")
    print(f"{'#'*60}")
    print(known_problems)

    model = build_fn(
        vocab_size=len(vocab),
        num_labels=len(label_encoder),
        embedding_matrix=embedding_matrix,
        hidden_dim=hidden_dim,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    checkpoint_path = f"{checkpoint_dir}/{model_key}_best.pt"

    # ── Skip / resume from checkpoint ─────────────────────────────────────
    if model_key in skip and Path(checkpoint_path).exists():
        print(f"[SKIP] Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        train_result = {
            "model_name":      display_name,
            "best_val_f1":     None,
            "best_val_recall": None,
            "best_epoch":      None,
            "total_time":      0,
            "per_epoch":       [],
            "gradient_problem": "Loaded from checkpoint — gradient history unavailable.",
            "checkpoint_path": checkpoint_path,
        }
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer   = Trainer(
            model          = model,
            optimizer      = optimizer,
            label_encoder  = label_encoder,
            model_name     = model_key,
            use_crf        = use_crf,
            patience       = patience,
            checkpoint_dir = checkpoint_dir,
        )
        train_result = trainer.train(train_loader, val_loader, epochs=epochs)
        train_result["model_name"] = display_name

    # ── Evaluate on test set ───────────────────────────────────────────────
    gold, pred = predict_sequences(model, test_loader, label_encoder, use_crf=use_crf)
    eval_result = full_eval(gold, pred, model_name=display_name)

    return train_result, eval_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2: Train and compare all 5 neural NER models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CPU training tips:
  Start with a limit to verify everything works before a full run:
    python run_stage2.py --data ... --limit 10000 --epochs 5

  Train one model at a time to checkpoint as you go:
    python run_stage2.py --data ... --limit 50000 --models rnn
    python run_stage2.py --data ... --limit 50000 --models lstm --skip rnn
    ...

  Estimated time per model on CPU with --limit 50000:
    RNN ~2h, LSTM ~3h, GRU ~2.5h, BiLSTM ~4h, BiLSTM+CRF ~5h
        """
    )
    parser.add_argument("--data",      required=True,
                        help="Path to dataset JSON file")
    parser.add_argument("--fasttext",  default=None,
                        help="Path to cc.hi.300.bin FastText model")
    parser.add_argument("--limit",     type=int, default=None,
                        help="Max examples to load. Strongly recommended on CPU "
                             "(e.g. --limit 50000 for demo-quality results).")
    parser.add_argument("--epochs",    type=int, default=20,
                        help="Max training epochs per model (default: 20 for CPU)")
    parser.add_argument("--patience",  type=int, default=3,
                        help="Early stopping patience (default: 3 for CPU)")
    parser.add_argument("--batch",     type=int, default=16,
                        help="Batch size (default: 16 for CPU; use 32+ on GPU)")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="RNN/LSTM/GRU hidden size (default: 128 for CPU; "
                             "use 256 on GPU)")
    parser.add_argument("--models",    nargs="+",
                        default=list(MODEL_REGISTRY.keys()),
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Which models to train (default: all)")
    parser.add_argument("--skip",      nargs="+", default=[],
                        help="Models to skip training (load checkpoint)")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="Directory for model checkpoints")
    parser.add_argument("--results-out", default=None,
                        help="Save results JSON to this path. "
                             "Defaults to results/<model>_<timestamp>.json")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── CPU warning ────────────────────────────────────────────────────────
    print(f"\n[INFO] Training on: CPU")
    if args.limit is None:
        print(
            "[WARN] No --limit set. Training on the full dataset on CPU will "
            "take many hours per model.\n"
            "       Recommended: --limit 50000 for good results, "
            "--limit 10000 for a quick test.\n"
        )
    else:
        print(f"[INFO] Using {args.limit:,} examples  |  "
              f"batch={args.batch}  hidden_dim={args.hidden_dim}  "
              f"epochs={args.epochs}  patience={args.patience}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    train_ex, val_ex, test_ex = load_and_split(
        args.data, limit=args.limit
    )
    vocab, label_encoder = build_vocab_and_labels(train_ex, min_freq=2)

    # ── Embeddings ────────────────────────────────────────────────────────
    embedding_matrix = load_embeddings(vocab, fasttext_path=args.fasttext)

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_loader = make_dataloader(train_ex, vocab, label_encoder,
                                   batch_size=args.batch, shuffle=True,
                                   num_workers=0)   # num_workers=0 required on CPU
    val_loader   = make_dataloader(val_ex,   vocab, label_encoder,
                                   batch_size=args.batch, shuffle=False,
                                   num_workers=0)
    test_loader  = make_dataloader(test_ex,  vocab, label_encoder,
                                   batch_size=args.batch, shuffle=False,
                                   num_workers=0)

    # ── Train all models ───────────────────────────────────────────────────
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    all_train_results = []
    all_eval_results  = []

    for model_key in args.models:
        train_result, eval_result = train_one_model(
            model_key        = model_key,
            vocab            = vocab,
            label_encoder    = label_encoder,
            embedding_matrix = embedding_matrix,
            train_loader     = train_loader,
            val_loader       = val_loader,
            test_loader      = test_loader,
            epochs           = args.epochs,
            patience         = args.patience,
            hidden_dim       = args.hidden_dim,
            checkpoint_dir   = args.checkpoint_dir,
            skip             = args.skip,
        )
        all_train_results.append(train_result)
        all_eval_results.append(eval_result)

    # ── Final comparison tables ────────────────────────────────────────────
    print_progression_table(all_eval_results)
    print_gradient_analysis(all_train_results)

    # ── JSON encoder that handles numpy types ─────────────────────────────
    class _JSONEncoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as np
            if isinstance(obj, np.integer):  return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray):  return obj.tolist()
            return super().default(obj)

    # ── Per-model timestamped file — never overwrites ──────────────────────
    from datetime import datetime
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_tag  = "_".join(args.models)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Full detail file: results/rnn_20250509_143201.json
    if args.results_out:
        out_path = Path(args.results_out)
    else:
        out_path = results_dir / f"{models_tag}_{timestamp}.json"

    output = {
        "timestamp":     timestamp,
        "models":        args.models,
        "limit":         args.limit,
        "epochs":        args.epochs,
        "hidden_dim":    args.hidden_dim,
        "train_results": all_train_results,
        "eval_results":  all_eval_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, cls=_JSONEncoder)
    print(f"\nResults saved to {out_path}")

    # ── Cumulative summary across all runs: results/summary.json ──────────
    # Appends one row per model — safe to run multiple times, never loses data
    summary_path = results_dir / "summary.json"
    summary = []
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    for eval_r in all_eval_results:
        summary.append({
            "timestamp":    timestamp,
            "model":        eval_r["model_name"],
            "limit":        args.limit,
            "epochs":       args.epochs,
            "hidden_dim":   args.hidden_dim,
            "macro_f1":     eval_r["macro_f1"],
            "macro_recall": eval_r["macro_recall"],
            "invalid_bio":  eval_r["invalid_bio"],
            "per_class":    eval_r.get("per_class", {}),
        })

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, cls=_JSONEncoder)
    print(f"Cumulative summary updated -> {summary_path}")


if __name__ == "__main__":
    main()
