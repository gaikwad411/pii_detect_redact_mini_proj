import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

epochs = [1, 2, 3, 4, 5, 6, 7]

data_50k = {
    "train": [0.6294, 0.6488, 0.6585, 0.6795, 0.6861, 0.7042, 0.7046],
    "val":   [0.5756, 0.5829, 0.5856, 0.5863, 0.5866, 0.5861, 0.5823],
    "gap":   [-0.0538, -0.0659, -0.0728, -0.0931, -0.0995, -0.1181, -0.1223],
}

data_100k = {
    "train": [0.6381, 0.6535, 0.6774, 0.6838, 0.6933, 0.6993, 0.7073],
    "val":   [0.5997, 0.6090, 0.6140, 0.6122, 0.6169, 0.6154, 0.6150],
    "gap":   [-0.0384, -0.0445, -0.0634, -0.0716, -0.0764, -0.0839, -0.0923],
}

C50  = "#e67e22"   # orange  — 50K
C100 = "#2980b9"   # blue    — 100K
BEST_ALPHA = 0.15

fig = plt.figure(figsize=(12, 9))
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

# ── helper ────────────────────────────────────────────────────────────────────
def style_ax(ax, title, ylabel, ylim=None):
    ax.set_title(title, fontsize=12, fontweight="bold", color="#2c3e50", pad=8)
    ax.set_xlabel("Epoch", fontsize=10, color="#555")
    ax.set_ylabel(ylabel, fontsize=10, color="#555")
    ax.set_xticks(epochs)
    ax.tick_params(colors="#555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#ccc")
    ax.spines["bottom"].set_color("#ccc")
    ax.grid(axis="y", color="#ececec", linewidth=0.8)
    if ylim:
        ax.set_ylim(*ylim)

def mark_best(ax, data, key, color):
    best_ep = max(range(len(epochs)), key=lambda i: data[key][i])
    ax.axvline(epochs[best_ep], color=color, linewidth=1.2, linestyle="--", alpha=0.6)
    ax.scatter([epochs[best_ep]], [data[key][best_ep]],
               color=color, s=80, zorder=5)

# ── 1. Val F1 comparison ──────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(epochs, data_50k["val"],  color=C50,  marker="o", linewidth=2, label="50K val F1")
ax1.plot(epochs, data_100k["val"], color=C100, marker="o", linewidth=2, label="100K val F1")
mark_best(ax1, data_50k,  "val", C50)
mark_best(ax1, data_100k, "val", C100)
ax1.legend(fontsize=9, framealpha=0.6)
style_ax(ax1, "Val F1 — 50K vs 100K", "Val F1", (0.55, 0.65))

# ── 2. Train F1 comparison ────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(epochs, data_50k["train"],  color=C50,  marker="s", linewidth=2, linestyle="--", label="50K train F1")
ax2.plot(epochs, data_100k["train"], color=C100, marker="s", linewidth=2, linestyle="--", label="100K train F1")
ax2.legend(fontsize=9, framealpha=0.6)
style_ax(ax2, "Train F1 — 50K vs 100K", "Train F1", (0.60, 0.73))

# ── 3. Overfitting gap ────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(epochs, data_50k["gap"],  color=C50,  marker="o", linewidth=2, label="50K gap")
ax3.plot(epochs, data_100k["gap"], color=C100, marker="o", linewidth=2, label="100K gap")
ax3.axhline(0, color="#aaa", linewidth=0.8, linestyle=":")
ax3.fill_between(epochs, data_50k["gap"],  0, color=C50,  alpha=0.08)
ax3.fill_between(epochs, data_100k["gap"], 0, color=C100, alpha=0.08)
ax3.legend(fontsize=9, framealpha=0.6)
style_ax(ax3, "Overfitting Gap (Train − Val F1)", "Gap", (-0.14, 0.01))
ax3.set_ylabel("Gap (negative = overfitting)", fontsize=9, color="#555")

# ── 4. Side-by-side best val + gap at epoch 5 ─────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
metrics = ["Best Val F1\n(epoch 5)", "Gap at\nepoch 5", "Gap at\nepoch 7"]
vals_50k  = [0.5866, abs(-0.0995), abs(-0.1223)]
vals_100k = [0.6169, abs(-0.0764), abs(-0.0923)]

x = range(len(metrics))
w = 0.32
bars1 = ax4.bar([i - w/2 for i in x], vals_50k,  width=w, color=C50,  label="50K",  alpha=0.85)
bars2 = ax4.bar([i + w/2 for i in x], vals_100k, width=w, color=C100, label="100K", alpha=0.85)

for bar in bars1:
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8, color=C50)
for bar in bars2:
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8, color=C100)

ax4.set_xticks(list(x))
ax4.set_xticklabels(metrics, fontsize=9)
ax4.legend(fontsize=9, framealpha=0.6)
style_ax(ax4, "Key Metrics Summary", "Value")

# ── title ─────────────────────────────────────────────────────────────────────
fig.suptitle("RNN — 50K vs 100K Training Examples", fontsize=14,
             fontweight="bold", color="#2c3e50", y=0.98)

out = Path("charts") / "rnn_50k_vs_100k.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out}")
