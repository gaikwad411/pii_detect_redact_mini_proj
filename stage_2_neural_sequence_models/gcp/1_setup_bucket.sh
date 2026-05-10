#!/bin/bash
# =============================================================================
# STEP 1 — One-time setup: create GCS bucket and upload dataset
#
# Run this ONCE from your local machine before provisioning VMs.
#
# Usage:
#   bash gcp/1_setup_bucket.sh
#
# Prerequisites:
#   gcloud auth login
#   gcloud config set project YOUR_PROJECT_ID
# =============================================================================

set -euo pipefail

# ── CONFIGURE THESE ───────────────────────────────────────────────────────────
PROJECT_ID="$(gcloud config get-value project)"
BUCKET="gs://${PROJECT_ID}-pii-training"
REGION="us-central1"
LOCAL_DATASET="$(dirname "$0")/../../../datasets/hi_IndicNER_v1.0/hi_train.json"
# ─────────────────────────────────────────────────────────────────────────────

echo "Project  : $PROJECT_ID"
echo "Bucket   : $BUCKET"
echo "Dataset  : $LOCAL_DATASET"
echo ""

# Create bucket if it doesn't exist
if gsutil ls "$BUCKET" &>/dev/null; then
    echo "[OK] Bucket already exists: $BUCKET"
else
    echo "[1/2] Creating bucket..."
    gsutil mb -p "$PROJECT_ID" -l "$REGION" "$BUCKET"
    echo "[OK] Bucket created: $BUCKET"
fi

# Upload dataset
echo "[2/2] Uploading dataset..."
gsutil cp "$LOCAL_DATASET" "$BUCKET/datasets/hi_train.json"
echo "[OK] Dataset uploaded to $BUCKET/datasets/hi_train.json"

echo ""
echo "Done. Set these in gcp/3_provision_vms.sh:"
echo "  BUCKET=$BUCKET"
echo "  PROJECT_ID=$PROJECT_ID"
