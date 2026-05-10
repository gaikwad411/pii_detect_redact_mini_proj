#!/bin/bash
# =============================================================================
# STEP 2 — VM startup script (runs automatically on each VM at boot)
#
# DO NOT run this manually. It is passed to GCP via --metadata-from-file.
# Each VM reads its assigned model from instance metadata.
#
# Flow:
#   install deps → clone repo → download dataset → install python deps
#   → train model → push results + log to GCS → shutdown VM
# =============================================================================

set -euo pipefail

WORKDIR="/opt/trainer"
LOGFILE="$WORKDIR/training.log"

mkdir -p "$WORKDIR"

# Redirect all output to log file AND serial console (visible in GCP Logs)
exec > >(tee -a "$LOGFILE") 2>&1

echo "============================================================"
echo "VM startup: $(date)"
echo "Hostname : $(hostname)"
echo "============================================================"


# ── Read config from instance metadata ───────────────────────────────────────
METADATA_URL="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
HEADER="Metadata-Flavor: Google"

MODEL=$(curl -sf "$METADATA_URL/model"   -H "$HEADER")
BUCKET=$(curl -sf "$METADATA_URL/bucket" -H "$HEADER")
LIMIT=$(curl -sf  "$METADATA_URL/limit"  -H "$HEADER" || echo "50000")
BATCH=$(curl -sf  "$METADATA_URL/batch"  -H "$HEADER" || echo "32")
PATIENCE=$(curl -sf "$METADATA_URL/patience" -H "$HEADER" || echo "5")

echo "Model    : $MODEL"
echo "Bucket   : $BUCKET"
echo "Limit    : $LIMIT"
echo "Batch    : $BATCH"
echo "Patience : $PATIENCE"
echo ""

REPO_DIR="$WORKDIR/repo"
STAGE2_DIR="$REPO_DIR/stage_2_neural_sequence_models"
DATASET_DIR="$REPO_DIR/datasets/hi_IndicNER_v1.0"
DATASET_FILE="$DATASET_DIR/hi_train.json"
RESULTS_DIR="$STAGE2_DIR/results"
EMBEDDINGS_DIR="$WORKDIR/embeddings"
FASTTEXT_FILE="$EMBEDDINGS_DIR/cc.hi.300.bin"


# ── 1. System dependencies ────────────────────────────────────────────────────
echo "[1/7] Installing system packages: $(date)"
apt-get update -qq
apt-get install -y -qq git curl build-essential


# ── 2. Install uv ─────────────────────────────────────────────────────────────
echo "[2/7] Installing uv: $(date)"
export HOME="$WORKDIR"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$WORKDIR/.local/bin:$PATH"
uv --version


# ── 3. Clone repo ─────────────────────────────────────────────────────────────
echo "[3/7] Cloning repo: $(date)"
git clone https://github.com/gaikwad411/pii_detect_redact_mini_proj.git "$REPO_DIR"


# ── 4. Download dataset from GCS ──────────────────────────────────────────────
echo "[4/7] Downloading dataset: $(date)"
mkdir -p "$DATASET_DIR"
gsutil cp "$BUCKET/datasets/hi_train.json" "$DATASET_FILE"
echo "Dataset size: $(du -sh $DATASET_FILE | cut -f1)"


# ── 5. Install Python dependencies ────────────────────────────────────────────
echo "[5/7] Installing Python dependencies: $(date)"
cd "$STAGE2_DIR"
uv sync
# Install fasttext separately — needs build-essential, compiles from C++ source.
# This works on Debian Linux but not on Windows (no pre-built wheels for Python 3.13).
uv pip install fasttext
echo "fasttext installed: $(uv run python -c 'import fasttext; print(fasttext.__version__)')"


# ── 6. Download FastText embeddings from GCS ─────────────────────────────────
# Upload once with: gsutil cp cc.en.300.bin gs://<bucket>/embeddings/cc.en.300.bin
echo "[6/7] Downloading FastText embeddings: $(date)"
mkdir -p "$EMBEDDINGS_DIR"
if [ -f "$FASTTEXT_FILE" ]; then
    echo "  Embeddings already present, skipping download."
else
    echo "  Downloading cc.hi.300.bin from $BUCKET/embeddings/ ..."
    gsutil cp "$BUCKET/embeddings/cc.hi.300.bin" "$FASTTEXT_FILE"
    echo "  Embeddings size: $(du -sh $FASTTEXT_FILE | cut -f1)"
fi


# ── 7. Run training ───────────────────────────────────────────────────────────
echo "[7/7] Training $MODEL: $(date)"

uv run run_stage2.py \
    --data      "$DATASET_FILE" \
    --models    "$MODEL" \
    --limit     "$LIMIT" \
    --batch     "$BATCH" \
    --patience  "$PATIENCE" \
    --num-workers 4 \
    --fasttext  "$FASTTEXT_FILE"

echo "Training complete: $(date)"


# ── 8. Push results to GCS ────────────────────────────────────────────────────
echo "Pushing results to GCS..."

# Copy all result JSON files for this model
gsutil cp "$RESULTS_DIR/${MODEL}_"*.json "$BUCKET/results/" 2>/dev/null || \
gsutil cp "$RESULTS_DIR/${MODEL}-"*.json "$BUCKET/results/" 2>/dev/null || true

# Always push summary
gsutil cp "$RESULTS_DIR/summary.json" "$BUCKET/results/summary_${MODEL}.json"

# Push log file
gsutil cp "$LOGFILE" "$BUCKET/logs/${MODEL}_training.log"

echo "Results pushed to $BUCKET/results/"
echo "Log pushed to $BUCKET/logs/${MODEL}_training.log"


# ── 8. Shutdown VM ────────────────────────────────────────────────────────────
echo "============================================================"
echo "All done: $(date)"
echo "Shutting down VM..."
echo "============================================================"
shutdown -h now
