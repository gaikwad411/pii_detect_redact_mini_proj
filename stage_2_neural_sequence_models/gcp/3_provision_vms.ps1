# STEP 3 - Provision one VM per model, all in parallel
#
# Run from your local Windows machine after step 1.
#
# Usage:
#   .\gcp\3_provision_vms.ps1                           # all 5 models
#   .\gcp\3_provision_vms.ps1 -Models "lstm","bilstm"   # specific models

param(
    [string[]]$Models = @("rnn","lstm","gru","bilstm","bilstm_crf")
)

$ErrorActionPreference = "Stop"

$PROJECT_ID     = (gcloud config get-value project)
$BUCKET         = "gs://$PROJECT_ID-pii-training"
$ZONE           = "us-central1-a"
$MACHINE_TYPE   = "e2-standard-4"   # 4 vCPU, fits within default quota (8 CPUs = 2 VMs at once)
$DISK_SIZE      = "30GB"
$IMAGE_FAMILY   = "debian-12"
$IMAGE_PROJECT  = "debian-cloud"
$STARTUP_SCRIPT = "$PSScriptRoot\2_vm_startup.sh"

# Training hyperparameters
$LIMIT    = 50000
$BATCH    = 32
$PATIENCE = 5

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
    $VM_NAME = "pii-trainer-$($MODEL -replace '_','-')"

    # Check if VM already exists
    $ErrorActionPreference = "Continue"
    gcloud compute instances describe $VM_NAME --zone=$ZONE --format="value(status)" 2>&1 | Out-Null
    $vmExists = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = "Stop"

    if ($vmExists) {
        Write-Host "[SKIP] $VM_NAME already exists - delete it first to re-run"
        continue
    }

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
        --async

    Write-Host "  Launched: $VM_NAME"
}

Write-Host ""
Write-Host "All VMs launched. Monitor with:"
Write-Host ""
Write-Host "  gcloud compute instances list --filter='name~pii-trainer'"
Write-Host ""
Write-Host "  # Stream log for a specific model"
Write-Host "  gcloud compute instances get-serial-port-output pii-trainer-lstm --zone=$ZONE"
Write-Host ""
Write-Host "  # Or check GCS log once training starts"
Write-Host "  gsutil cat $BUCKET/logs/lstm_training.log"
Write-Host ""
Write-Host "Run step 4 once all VMs shut down:"
Write-Host "  .\gcp\4_collect_results.ps1"
