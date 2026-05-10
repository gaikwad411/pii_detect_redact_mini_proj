"""
plot_training.py
----------------
Generate training charts for all Stage 2 models.

Charts produced:
  1. Val F1 across epochs — all 5 models on one chart
  2. Training loss across epochs — all 5 models on one chart
  3. Train F1 vs Val F1 per model — overfitting gap (only if train_f1 exists)
  4. Final test metrics bar chart — F1 / Recall / Invalid BIO

Usage
-----
    python plot_training.py                         # reads results/
    python plot_training.py --results-dir results   # explicit path
    python plot_training.py --out charts            # save to charts/
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no display needed — saves to file
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Style ──────────────────────────────────────────────────────────────────
MODEL_ORDER  = ["RNN", "LSTM", "GRU", "BiLSTM", "BiLSTM+CRF"]
MODEL_COLORS = {
    "RNN":        "#e74c3c",
    "LSTM":       "#e67e22",
    "GRU":        "#f1c40f",
    "BiLSTM":     "#2ecc71",
    "BiLSTM+CRF": "#2980b9",
}
MODEL_MARKERS = {
    "RNN": "o", "LSTM": "s", "GRU": "^", "BiLSTM": "D", "BiLSTM+CRF": "*",
}

plt.rcParams.update({
    "figure.dpi":      150,
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  11,
    "legend.fontsize": 10,
    "lines.linewidth": 2,
    "lines.markersize": 6,
})


# ── Data loading ────────────────────────────────────────────────────────────

def load_results(results_dir: Path) -> dict[str, dict]:
    """
    Load the latest result file per model from results_dir.
    Returns {model_name: result_dict}.
    """
    latest: dict[str, tuple[str, dict]] = {}

    for f in sorted(results_dir.glob("*.json")):
        if f.name == "summary.json":
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue

        # Individual model file: {"models": ["rnn"], "train_results": [...], ...}
        if "train_results" not in data:
            continue

        for tr in data["train_results"]:
            name = tr["model_name"].upper().replace("_", "+").replace("BILSTM+CRF", "BiLSTM+CRF")
            # Normalise name
            for canonical in MODEL_ORDER:
                if name.upper() == canonical.upper():
                    name = canonical
                    break

            ts = data.get("timestamp", f.stem)
            if name not in latest or ts > latest[name][0]:
                latest[name] = (ts, {"train": tr, "eval": None})

        for ev in data.get("eval_results", []):
            name = ev["model_name"].upper().replace("_", "+")
            for canonical in MODEL_ORDER:
                if name.upper() == canonical.upper():
                    name = canonical
                    break
            if name in latest:
                latest[name][1]["eval"] = ev

    return {k: v[1] for k, v in latest.items()}


# ── Chart 1: Val F1 across epochs ──────────────────────────────────────────

def plot_val_f1(results: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for name in MODEL_ORDER:
        if name not in results:
            continue
        epochs_data = results[name]["train"]["per_epoch"]
        xs = [r["epoch"] for r in epochs_data]
        ys = [r["val_f1"] for r in epochs_data]
        best_epoch = results[name]["train"]["best_epoch"]

        ax.plot(xs, ys, color=MODEL_COLORS[name], marker=MODEL_MARKERS[name],
                label=name, markevery=[best_epoch - 1])
        # Mark best epoch with a star
        ax.scatter([best_epoch], [ys[best_epoch - 1]],
                   color=MODEL_COLORS[name], s=120, zorder=5,
                   edgecolors="black", linewidths=0.8)

    ax.set_title("Validation F1 across Epochs — Stage 2 Models")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro F1 (validation)")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    path = out_dir / "01_val_f1_epochs.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Chart 2: Training loss across epochs ───────────────────────────────────

def plot_loss(results: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for name in MODEL_ORDER:
        if name not in results:
            continue
        if name == "BiLSTM+CRF":
            continue   # CRF loss is on a different scale (100s vs 0.x)
        epochs_data = results[name]["train"]["per_epoch"]
        xs = [r["epoch"] for r in epochs_data]
        ys = [r["loss"]   for r in epochs_data]
        ax.plot(xs, ys, color=MODEL_COLORS[name], marker=MODEL_MARKERS[name], label=name)

    ax.set_title("Training Loss across Epochs (RNN / LSTM / GRU / BiLSTM)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss (training set)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    path = out_dir / "02_train_loss_epochs.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")

    # Separate chart for BiLSTM+CRF (different loss scale)
    if "BiLSTM+CRF" in results:
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        epochs_data = results["BiLSTM+CRF"]["train"]["per_epoch"]
        xs = [r["epoch"] for r in epochs_data]
        ys = [r["loss"]   for r in epochs_data]
        ax2.plot(xs, ys, color=MODEL_COLORS["BiLSTM+CRF"],
                 marker=MODEL_MARKERS["BiLSTM+CRF"], label="BiLSTM+CRF")
        ax2.set_title("BiLSTM+CRF Training Loss (sequence-level NLL)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Negative log-likelihood")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        path2 = out_dir / "02b_bilstm_crf_loss.png"
        fig2.tight_layout()
        fig2.savefig(path2)
        plt.close(fig2)
        print(f"  Saved: {path2}")


# ── Chart 3: Train F1 vs Val F1 (overfitting gap) ──────────────────────────

def plot_overfit_gap(results: dict, out_dir: Path) -> None:
    # Only generate if train_f1 data exists
    models_with_train_f1 = [
        name for name in MODEL_ORDER
        if name in results
        and any("train_f1" in r for r in results[name]["train"]["per_epoch"])
    ]

    if not models_with_train_f1:
        print("  [SKIP] Train F1 not in results — retrain with updated trainer.py first.")
        return

    n = len(models_with_train_f1)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, models_with_train_f1):
        epochs_data = results[name]["train"]["per_epoch"]
        xs       = [r["epoch"]    for r in epochs_data]
        train_f1 = [r["train_f1"] for r in epochs_data]
        val_f1   = [r["val_f1"]   for r in epochs_data]
        best_ep  = results[name]["train"]["best_epoch"]

        ax.plot(xs, train_f1, color=MODEL_COLORS[name], linestyle="-",
                marker="o", label="Train F1")
        ax.plot(xs, val_f1,   color=MODEL_COLORS[name], linestyle="--",
                marker="s", label="Val F1", alpha=0.7)

        # Shade the gap
        ax.fill_between(xs, val_f1, train_f1, alpha=0.12,
                        color=MODEL_COLORS[name], label="Overfit gap")

        # Mark best val epoch
        ax.axvline(best_ep, color="gray", linestyle=":", alpha=0.7,
                   label=f"Best epoch ({best_ep})")

        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Macro F1")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Train F1 vs Val F1 — Overfitting Gap", fontsize=13, y=1.02)
    path = out_dir / "03_train_vs_val_f1.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Chart 4: Final test metrics bar chart ──────────────────────────────────

def plot_final_metrics(results: dict, out_dir: Path) -> None:
    names   = [n for n in MODEL_ORDER if n in results and results[n]["eval"]]
    f1s     = [results[n]["eval"]["macro_f1"]     for n in names]
    recalls = [results[n]["eval"]["macro_recall"]  for n in names]
    colors  = [MODEL_COLORS[n] for n in names]

    x = range(len(names))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # F1 and Recall grouped bar
    bars1 = ax1.bar([i - width/2 for i in x], f1s,     width, label="Macro F1",     color=colors, alpha=0.85)
    bars2 = ax1.bar([i + width/2 for i in x], recalls,  width, label="Macro Recall", color=colors, alpha=0.5, edgecolor="black", linewidth=0.8)

    ax1.set_title("Final Test: Macro F1 and Recall")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(names, rotation=15, ha="right")
    ax1.set_ylim(0, 1.0)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)

    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    # Invalid BIO bar
    inv_bios = [results[n]["eval"]["invalid_bio"] for n in names]
    ax2.bar(names, inv_bios, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax2.set_title("Invalid BIO Sequences (target = 0)")
    ax2.set_ylabel("Count")
    ax2.set_xticklabels(names, rotation=15, ha="right")
    ax2.grid(True, axis="y", alpha=0.3)

    for i, v in enumerate(inv_bios):
        ax2.text(i, v + 10, str(v), ha="center", va="bottom", fontsize=9)

    fig.suptitle("Stage 2 — Final Test Evaluation", fontsize=13)
    path = out_dir / "04_final_test_metrics.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Chart 5: Per-class recall progression ──────────────────────────────────

def plot_per_class_recall(results: dict, out_dir: Path) -> None:
    names = [n for n in MODEL_ORDER if n in results and results[n]["eval"]]
    classes = ["PER", "ORG", "LOC"]
    class_colors = {"PER": "#3498db", "ORG": "#e74c3c", "LOC": "#2ecc71"}

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(names))
    width = 0.25

    for i, cls in enumerate(classes):
        recalls = [
            results[n]["eval"]["per_class"].get(cls, {}).get("recall", 0)
            for n in names
        ]
        offset = (i - 1) * width
        bars = ax.bar([xi + offset for xi in x], recalls, width,
                      label=cls, color=class_colors[cls], alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_title("Per-Class Recall Progression across Models")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    path = out_dir / "05_per_class_recall.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot Stage 2 training charts")
    parser.add_argument("--results-dir", default="results",
                        help="Directory with result JSON files (default: results)")
    parser.add_argument("--out", default="charts",
                        help="Output directory for PNG files (default: charts)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    print(f"Loading results from: {results_dir}")
    results = load_results(results_dir)

    if not results:
        print("[ERROR] No result files found. Run run_stage2.py first.")
        return

    print(f"Models found: {list(results.keys())}")
    print(f"Saving charts to: {out_dir}/\n")

    plot_val_f1(results, out_dir)
    plot_loss(results, out_dir)
    plot_overfit_gap(results, out_dir)
    plot_final_metrics(results, out_dir)
    plot_per_class_recall(results, out_dir)

    print(f"\nDone. {len(list(out_dir.glob('*.png')))} charts saved to {out_dir}/")


if __name__ == "__main__":
    main()
