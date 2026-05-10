# =============================================================================
# STEP 1 — One-time setup: create GCS bucket and upload dataset
#
# Run ONCE from your local Windows machine before provisioning VMs.
#
# Prerequisites:
#   gcloud auth login
#   gcloud config set project YOUR_PROJECT_ID
# =============================================================================

$ErrorActionPreference = "Stop"

# ── CONFIGURE THESE ───────────────────────────────────────────────────────────
$PROJECT_ID   = (gcloud config get-value project)
$BUCKET       = "gs://$PROJECT_ID-pii-training"
$REGION       = "us-central1"
$LOCAL_DATASET = "$PSScriptRoot\..\..\..\datasets\hi_IndicNER_v1.0\hi_train.json"
# ─────────────────────────────────────────────────────────────────────────────

Write-Host "Project  : $PROJECT_ID"
Write-Host "Bucket   : $BUCKET"
Write-Host "Dataset  : $LOCAL_DATASET"
Write-Host ""

if (-not (Test-Path $LOCAL_DATASET)) {
    Write-Error "Dataset not found at: $LOCAL_DATASET"
    exit 1
}

# Create bucket if it doesn't exist
$bucketExists = (gsutil ls $BUCKET 2>$null) -ne $null
if ($bucketExists) {
    Write-Host "[OK] Bucket already exists: $BUCKET"
} else {
    Write-Host "[1/2] Creating bucket..."
    gsutil mb -p $PROJECT_ID -l $REGION $BUCKET
    Write-Host "[OK] Bucket created: $BUCKET"
}

# Upload dataset
Write-Host "[2/2] Uploading dataset..."
gsutil cp $LOCAL_DATASET "$BUCKET/datasets/hi_train.json"
Write-Host "[OK] Uploaded to $BUCKET/datasets/hi_train.json"

Write-Host ""
Write-Host "Done. Your bucket name for the next steps:"
Write-Host "  BUCKET = $BUCKET"
