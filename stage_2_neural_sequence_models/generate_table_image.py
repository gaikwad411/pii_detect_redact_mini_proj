import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

rows = [
    ("Model",       "Approach",    "Macro F1", "Recall",  "Inv BIO"),  # header
    ("Regex",       "Rule-based",  "0.2183",   "0.1591",  "N/A"),
    ("RNN",         "Neural",      "0.5878",   "0.6101",  "1803"),
    ("LSTM",        "Neural",      "0.6003",   "0.6082",  "1541"),
    ("GRU",         "Neural",      "0.5978",   "0.6111",  "1435"),
    ("BiLSTM",      "Neural",      "0.6778",   "0.6809",  "535"),
    ("BiLSTM+CRF",  "Neural",      "0.7065",   "0.6818",  "0"),
]

col_widths = [0.18, 0.18, 0.16, 0.16, 0.14]
col_x      = [0.04, 0.22, 0.40, 0.56, 0.72]
row_h      = 0.11
fig_w, fig_h = 8, 3.6

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Colours
HDR_BG   = "#2c3e50"
HDR_FG   = "white"
REG_BG   = "#f0e6ff"   # light purple for regex row
EVEN_BG  = "#f8f9fa"
ODD_BG   = "white"
BEST_BG  = "#d4efdf"   # light green for BiLSTM+CRF
LINE_COL = "#bdc3c7"

n_rows = len(rows)
top = 0.95

for r_idx, row in enumerate(rows):
    y_top = top - r_idx * row_h

    # Row background
    if r_idx == 0:
        bg = HDR_BG
    elif r_idx == 1:
        bg = REG_BG
    elif r_idx == n_rows - 1:
        bg = BEST_BG
    else:
        bg = EVEN_BG if r_idx % 2 == 0 else ODD_BG

    rect = mpatches.FancyBboxPatch(
        (0.02, y_top - row_h + 0.005), 0.96, row_h - 0.008,
        boxstyle="round,pad=0.005",
        linewidth=0,
        facecolor=bg,
        transform=ax.transAxes,
        zorder=1
    )
    ax.add_patch(rect)

    for c_idx, (cell, cx, cw) in enumerate(zip(row, col_x, col_widths)):
        fg = HDR_FG if r_idx == 0 else "#2c3e50"
        weight = "bold" if r_idx == 0 or r_idx == n_rows - 1 else "normal"
        align = "left" if c_idx <= 1 else "center"
        x_pos = cx + 0.01 if align == "left" else cx + cw / 2

        ax.text(
            x_pos, y_top - row_h / 2,
            cell,
            ha=align, va="center",
            fontsize=11, fontweight=weight,
            color=fg,
            transform=ax.transAxes,
            zorder=2
        )

# Divider line after header
y_div = top - row_h + 0.002
ax.plot([0.02, 0.98], [y_div, y_div],
        color=HDR_BG, linewidth=1.5, transform=ax.transAxes, zorder=3)

# Divider line after regex row
y_div2 = top - 2 * row_h + 0.002
ax.plot([0.02, 0.98], [y_div2, y_div2],
        color=LINE_COL, linewidth=1, transform=ax.transAxes, zorder=3)

# Outer border
outer = mpatches.FancyBboxPatch(
    (0.02, top - n_rows * row_h + 0.005), 0.96, n_rows * row_h - 0.008,
    boxstyle="round,pad=0.005",
    linewidth=1.5,
    edgecolor=HDR_BG,
    facecolor="none",
    transform=ax.transAxes,
    zorder=4
)
ax.add_patch(outer)

out = Path("charts") / "results_table.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out}")
