#!/bin/bash
# =============================================================================
# GCP Instance Startup Script
# Stage 2 — Neural NER Training
#
# How to use:
#   Option A — paste into "Startup script" field in GCP Console when
#              creating the VM (Compute Engine → Create Instance →
#              Advanced → Automation → Startup script)
#
#   Option B — attach via gcloud CLI:
#     gcloud compute instances create stage2-trainer \
#       --machine-type=c3-highcpu-8 \
#       --zone=us-central1-a \
#       --metadata-from-file startup-script=gcp_startup.sh \
#       --boot-disk-size=50GB \
#       --image-family=debian-12 \
#       --image-project=debian-cloud
#
# Dataset:
#   The datasets/ folder is gitignored.
#   Set GCS_DATASET_URI below to your GCS path, e.g.:
#     gs://your-bucket/hi_IndicNER_v1.0/hi_train.json
#   Then upload your dataset once:
#     gsutil cp path/to/hi_train.json gs://your-bucket/hi_IndicNER_v1.0/
#
# Logs:
#   All output goes to /home/trainer/training.log
#   Tail it live via:  gcloud compute ssh <vm-name> -- tail -f /home/trainer/training.log
# =============================================================================

set -euo pipefail

# ── CONFIGURE THESE ───────────────────────────────────────────────────────────
GITHUB_REPO="https://github.com/gaikwad411/pii_detect_redact_mini_proj.git"
GCS_DATASET_URI="gs://YOUR_BUCKET/hi_IndicNER_v1.0/hi_train.json"   # ← set this
MODELS="lstm gru bilstm bilstm_crf"   # models to train (rnn already done locally)
LIMIT=50000
PATIENCE=5
BATCH=32
NUM_WORKERS=4
# ─────────────────────────────────────────────────────────────────────────────

WORKDIR="/home/trainer"
LOGFILE="$WORKDIR/training.log"
REPO_DIR="$WORKDIR/pii_detect_redact_mini_proj"
STAGE2_DIR="$REPO_DIR/stage_2_neural_sequence_models"
DATASET_DIR="$REPO_DIR/datasets/hi_IndicNER_v1.0"

# Redirect all stdout and stderr to log file AND console
exec > >(tee -a "$LOGFILE") 2>&1

echo "============================================================"
echo "Startup script begin: $(date)"
echo "============================================================"


# ── 1. System dependencies ────────────────────────────────────────────────────
echo "[1/6] Installing system packages..."
apt-get update -qq
apt-get install -y -qq git curl build-essential


# ── 2. Install uv ─────────────────────────────────────────────────────────────
echo "[2/6] Installing uv..."
export HOME="$WORKDIR"
mkdir -p "$WORKDIR"

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$WORKDIR/.cargo/bin:$WORKDIR/.local/bin:$PATH"

uv --version


# ── 3. Clone repo ─────────────────────────────────────────────────────────────
echo "[3/6] Cloning repo..."
if [ -d "$REPO_DIR" ]; then
    echo "  Repo already exists, pulling latest..."
    git -C "$REPO_DIR" pull
else
    git clone "$GITHUB_REPO" "$REPO_DIR"
fi


# ── 4. Download dataset from GCS ──────────────────────────────────────────────
echo "[4/6] Downloading dataset..."
mkdir -p "$DATASET_DIR"

DATASET_FILE="$DATASET_DIR/hi_train.json"
if [ -f "$DATASET_FILE" ]; then
    echo "  Dataset already present, skipping download."
else
    echo "  Downloading from $GCS_DATASET_URI ..."
    gsutil cp "$GCS_DATASET_URI" "$DATASET_FILE"
fi


# ── 5. Install Python dependencies ────────────────────────────────────────────
echo "[5/6] Installing Python dependencies..."
cd "$STAGE2_DIR"
uv sync


# ── 6. Run training ───────────────────────────────────────────────────────────
echo "[6/6] Starting training: $(date)"
echo "  Models   : $MODELS"
echo "  Limit    : $LIMIT"
echo "  Patience : $PATIENCE"
echo "  Batch    : $BATCH"
echo "  Workers  : $NUM_WORKERS"
echo ""

cd "$STAGE2_DIR"

for MODEL in $MODELS; do
    echo "------------------------------------------------------------"
    echo "Training model: $MODEL  |  $(date)"
    echo "------------------------------------------------------------"

    uv run run_stage2.py \
        --data "$DATASET_FILE" \
        --models "$MODEL" \
        --limit  "$LIMIT" \
        --patience "$PATIENCE" \
        --batch  "$BATCH" \
        --num-workers "$NUM_WORKERS"

    echo "Done: $MODEL  |  $(date)"
done

echo "============================================================"
echo "All training complete: $(date)"
echo "Results saved to: $STAGE2_DIR/results/"
echo "============================================================"

# Optional: shut down the VM automatically when training is done
# (saves cost — comment out if you want the VM to stay up)
# shutdown -h now
