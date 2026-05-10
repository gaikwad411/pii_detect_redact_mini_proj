#!/bin/bash
# =============================================================================
# STEP 3 — Provision one VM per model, all in parallel
#
# Run from your local machine after step 1.
#
# Usage:
#   bash gcp/3_provision_vms.sh
#   bash gcp/3_provision_vms.sh lstm bilstm    # provision specific models only
#
# Each VM:
#   - Named: pii-trainer-<model>  (e.g. pii-trainer-bilstm-crf)
#   - Starts training automatically via startup script
#   - Shuts itself down when done
# =============================================================================

set -euo pipefail

# ── CONFIGURE THESE ───────────────────────────────────────────────────────────
PROJECT_ID="$(gcloud config get-value project)"
BUCKET="gs://${PROJECT_ID}-pii-training"
ZONE="us-central1-a"
MACHINE_TYPE="c3-highcpu-8"
DISK_SIZE="30GB"
IMAGE_FAMILY="debian-12"
IMAGE_PROJECT="debian-cloud"
STARTUP_SCRIPT="$(dirname "$0")/2_vm_startup.sh"

# Training hyperparameters passed to each VM via metadata
LIMIT=50000
BATCH=32
PATIENCE=5
# ─────────────────────────────────────────────────────────────────────────────

# Models to train — pass as args or train all
ALL_MODELS="rnn lstm gru bilstm bilstm_crf"
MODELS="${*:-$ALL_MODELS}"

echo "============================================================"
echo "Provisioning VMs"
echo "Project      : $PROJECT_ID"
echo "Bucket       : $BUCKET"
echo "Zone         : $ZONE"
echo "Machine type : $MACHINE_TYPE"
echo "Models       : $MODELS"
echo "============================================================"
echo ""

for MODEL in $MODELS; do
    # GCP VM names must be lowercase with hyphens only
    VM_NAME="pii-trainer-$(echo $MODEL | tr '_' '-')"

    echo "Creating VM: $VM_NAME  (model=$MODEL)"

    gcloud compute instances create "$VM_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --image-family="$IMAGE_FAMILY" \
        --image-project="$IMAGE_PROJECT" \
        --boot-disk-size="$DISK_SIZE" \
        --boot-disk-type="pd-ssd" \
        --scopes="storage-rw,logging-write,monitoring-write" \
        --metadata="model=$MODEL,bucket=$BUCKET,limit=$LIMIT,batch=$BATCH,patience=$PATIENCE" \
        --metadata-from-file="startup-script=$STARTUP_SCRIPT" \
        --no-address \
        --async

    echo "  Launched: $VM_NAME"
done

echo ""
echo "All VMs launched. Monitor with:"
echo ""
echo "  # List VM status"
echo "  gcloud compute instances list --filter='name~pii-trainer'"
echo ""
echo "  # Stream log for a specific model (e.g. lstm)"
echo "  gcloud compute instances get-serial-port-output pii-trainer-lstm --zone=$ZONE"
echo ""
echo "  # Or tail log from GCS once it starts uploading"
echo "  gsutil cat $BUCKET/logs/lstm_training.log"
echo ""
echo "Run step 4 once VMs finish (they shut down automatically):"
echo "  bash gcp/4_collect_results.sh"
