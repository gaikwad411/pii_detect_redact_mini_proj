# STEP 4 - Download results and logs from GCS to local machine
# Run after VMs have finished (they auto-shutdown when training is done).
#
# Usage:
#   .\gcp\4_collect_results.ps1
#
# Downloads to:
#   results\gcp\        <- result JSON files
#   results\gcp\logs\   <- stdout logs per model

$ErrorActionPreference = "Stop"

$PROJECT_ID    = (gcloud config get-value project)
$BUCKET        = "gs://$PROJECT_ID-pii-training"
$STAGE2_DIR    = (Get-Item "$PSScriptRoot\..").FullName
$LOCAL_RESULTS = "$STAGE2_DIR\results\gcp"
$LOCAL_LOGS    = "$LOCAL_RESULTS\logs"

New-Item -ItemType Directory -Force -Path $LOCAL_RESULTS | Out-Null
New-Item -ItemType Directory -Force -Path $LOCAL_LOGS    | Out-Null

Write-Host "============================================================"
Write-Host "Collecting results from GCS"
Write-Host "Bucket : $BUCKET"
Write-Host "Into   : $LOCAL_RESULTS"
Write-Host "============================================================"
Write-Host ""

# Check which VMs are still running
Write-Host "Checking VM status..."
$RUNNING = gcloud compute instances list --filter="name~pii-trainer AND status=RUNNING" --format="value(name)" 2>&1

if ($RUNNING -and $RUNNING -notmatch "Listed 0") {
    Write-Warning "These VMs are still running - results may be incomplete:"
    Write-Host $RUNNING
    Write-Host ""
    $confirm = Read-Host "Continue downloading anyway? [y/N]"
    if ($confirm -notmatch "^[Yy]$") { exit 0 }
} else {
    Write-Host "  All VMs are stopped. Good to collect."
}

Write-Host ""

# Download result JSONs
Write-Host "[1/3] Downloading result JSON files..."
try {
    gsutil -m cp "$BUCKET/results/*.json" "$LOCAL_RESULTS\" 2>&1 | Out-Null
    Write-Host "  Done"
} catch {
    Write-Warning "No result files found yet in $BUCKET/results/"
}

# Download logs
Write-Host "[2/3] Downloading training logs..."
try {
    gsutil -m cp "$BUCKET/logs/*.log" "$LOCAL_LOGS\" 2>&1 | Out-Null
    Write-Host "  Done"
} catch {
    Write-Warning "No log files found yet in $BUCKET/logs/"
}

# Merge all summary_<model>.json into one summary_merged.json
Write-Host "[3/3] Merging summary files..."
$summaryFiles = Get-ChildItem "$LOCAL_RESULTS\summary_*.json" -ErrorAction SilentlyContinue

if ($summaryFiles -and $summaryFiles.Count -gt 0) {
    $merged = @()
    $seen   = @{}

    foreach ($file in $summaryFiles) {
        $rows = Get-Content $file.FullName -Raw | ConvertFrom-Json
        foreach ($row in $rows) {
            $key = "$($row.model)_$($row.timestamp)"
            if (-not $seen.ContainsKey($key)) {
                $seen[$key] = $true
                $merged    += $row
            }
        }
    }

    $outPath = "$LOCAL_RESULTS\summary_merged.json"
    $merged | ConvertTo-Json -Depth 10 | Set-Content $outPath -Encoding utf8
    Write-Host "  Merged $($summaryFiles.Count) files -> $($merged.Count) rows -> $outPath"
} else {
    Write-Warning "No summary files to merge yet"
}

Write-Host ""
Write-Host "============================================================"
Write-Host "Downloaded to: $LOCAL_RESULTS"
Write-Host ""
Get-ChildItem $LOCAL_RESULTS | Format-Table Name, Length, LastWriteTime
Write-Host ""
Write-Host "Next - generate comparison table and charts:"
Write-Host "  uv run compare_models.py --summary results\gcp\summary_merged.json"
Write-Host "  uv run plot_training.py --results-dir results\gcp --out charts\gcp"
Write-Host "============================================================"
