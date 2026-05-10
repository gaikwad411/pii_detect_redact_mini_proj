# =============================================================================
# STEP 3 — Provision one VM per model, all in parallel
#
# Run from your local Windows machine after step 1.
#
# Usage:
#   .\gcp\3_provision_vms.ps1                           # all 5 models
#   .\gcp\3_provision_vms.ps1 -Models "lstm","bilstm"   # specific models
# =============================================================================

param(
    [string[]]$Models = @("rnn","lstm","gru","bilstm","bilstm_crf")
)

$ErrorActionPreference = "Stop"

# ── CONFIGURE THESE ───────────────────────────────────────────────────────────
$PROJECT_ID     = (gcloud config get-value project)
$BUCKET         = "gs://$PROJECT_ID-pii-training"
$ZONE           = "us-central1-a"
$MACHINE_TYPE   = "c3-highcpu-8"
$DISK_SIZE      = "30GB"
$IMAGE_FAMILY   = "debian-12"
$IMAGE_PROJECT  = "debian-cloud"
$STARTUP_SCRIPT = "$PSScriptRoot\2_vm_startup.sh"

# Training hyperparameters
$LIMIT    = 50000
$BATCH    = 32
$PATIENCE = 5
# ─────────────────────────────────────────────────────────────────────────────

Write-Host "============================================================"
Write-Host "Provisioning VMs"
Write-Host "Project      : $PROJECT_ID"
Write-Host "Bucket       : $BUCKET"
Write-Host "Zone         : $ZONE"
Write-Host "Machine type : $MACHINE_TYPE"
Write-Host "Models       : $($Models -join ', ')"
Write-Host "============================================================"
Write-Host ""

foreach ($MODEL in $Models) {
    # GCP VM names must be lowercase with hyphens only
    $VM_NAME = "pii-trainer-$($MODEL -replace '_','-')"

    Write-Host "Creating VM: $VM_NAME  (model=$MODEL)"

    gcloud compute instances create $VM_NAME `
        --project=$PROJECT_ID `
        --zone=$ZONE `
        --machine-type=$MACHINE_TYPE `
        --image-family=$IMAGE_FAMILY `
        --image-project=$IMAGE_PROJECT `
        --boot-disk-size=$DISK_SIZE `
        --boot-disk-type=pd-ssd `
        --scopes="storage-rw,logging-write,monitoring-write" `
        --metadata="model=$MODEL,bucket=$BUCKET,limit=$LIMIT,batch=$BATCH,patience=$PATIENCE" `
        --metadata-from-file="startup-script=$STARTUP_SCRIPT" `
        --no-address `
        --async

    Write-Host "  Launched: $VM_NAME"
}

Write-Host ""
Write-Host "All VMs launched. Monitor with:"
Write-Host ""
Write-Host "  # List VM status"
Write-Host "  gcloud compute instances list --filter='name~pii-trainer'"
Write-Host ""
Write-Host "  # Stream serial console log for a specific model"
Write-Host "  gcloud compute instances get-serial-port-output pii-trainer-lstm --zone=$ZONE"
Write-Host ""
Write-Host "  # Check log in GCS once training starts"
Write-Host "  gsutil cat $BUCKET/logs/lstm_training.log"
Write-Host ""
Write-Host "Run step 4 once all VMs shut down:"
Write-Host "  .\gcp\4_collect_results.ps1"
