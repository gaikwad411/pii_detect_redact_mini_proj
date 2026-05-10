#!/bin/bash
# =============================================================================
# STEP 4 — Download results and logs from GCS back to local machine
#
# Run after VMs have finished (they auto-shutdown when done).
#
# Usage:
#   bash gcp/4_collect_results.sh
#
# Downloads to:
#   results/gcp/          ← result JSON files
#   results/gcp/logs/     ← training logs (stdout) per model
# =============================================================================

set -euo pipefail

# ── CONFIGURE THESE ───────────────────────────────────────────────────────────
PROJECT_ID="$(gcloud config get-value project)"
BUCKET="gs://${PROJECT_ID}-pii-training"
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE2_DIR="$(dirname "$SCRIPT_DIR")"
LOCAL_RESULTS="$STAGE2_DIR/results/gcp"
LOCAL_LOGS="$LOCAL_RESULTS/logs"

mkdir -p "$LOCAL_RESULTS" "$LOCAL_LOGS"

echo "============================================================"
echo "Collecting results from GCS"
echo "Bucket : $BUCKET"
echo "Into   : $LOCAL_RESULTS"
echo "============================================================"
echo ""

# Check which VMs are still running
echo "Checking VM status..."
RUNNING=$(gcloud compute instances list \
    --filter="name~pii-trainer AND status=RUNNING" \
    --format="value(name)" 2>/dev/null || true)

if [ -n "$RUNNING" ]; then
    echo "[WARN] These VMs are still running — results may be incomplete:"
    echo "$RUNNING" | sed 's/^/  /'
    echo ""
    read -p "Continue downloading anyway? [y/N] " -n 1 -r
    echo ""
    [[ $REPLY =~ ^[Yy]$ ]] || exit 0
fi

# Download result JSONs
echo "[1/3] Downloading result files..."
gsutil -m cp "$BUCKET/results/*.json" "$LOCAL_RESULTS/" 2>/dev/null \
    && echo "  Done" \
    || echo "  [WARN] No result files found yet"

# Download logs
echo "[2/3] Downloading training logs..."
gsutil -m cp "$BUCKET/logs/*.log" "$LOCAL_LOGS/" 2>/dev/null \
    && echo "  Done" \
    || echo "  [WARN] No log files found yet"

# Merge summary files from all models into one
echo "[3/3] Merging summary files..."
SUMMARIES=("$LOCAL_RESULTS"/summary_*.json)
if [ ${#SUMMARIES[@]} -gt 0 ] && [ -f "${SUMMARIES[0]}" ]; then
    python3 - <<'EOF'
import json, glob, sys
from pathlib import Path

results_dir = Path(sys.argv[1])
summaries = list(results_dir.glob("summary_*.json"))

merged = []
seen = set()
for f in sorted(summaries):
    rows = json.loads(f.read_text())
    if isinstance(rows, list):
        for row in rows:
            key = (row.get("model"), row.get("timestamp"))
            if key not in seen:
                seen.add(key)
                merged.append(row)

out = results_dir / "summary_merged.json"
out.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
print(f"  Merged {len(summaries)} summary files → {len(merged)} rows → {out}")
EOF
    python3 -c "
import json, glob, sys
from pathlib import Path
results_dir = Path('$LOCAL_RESULTS')
summaries = list(results_dir.glob('summary_*.json'))
merged = []
seen = set()
for f in sorted(summaries):
    rows = json.loads(f.read_text())
    if isinstance(rows, list):
        for row in rows:
            key = (row.get('model'), row.get('timestamp'))
            if key not in seen:
                seen.add(key)
                merged.append(row)
out = results_dir / 'summary_merged.json'
out.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
print(f'  Merged {len(summaries)} files → {len(merged)} rows → {out}')
"
else
    echo "  No summary files to merge yet"
fi

echo ""
echo "============================================================"
echo "Local results: $LOCAL_RESULTS"
echo ""
ls -lh "$LOCAL_RESULTS" 2>/dev/null || true
echo ""
echo "Next: run compare_models.py against GCP results:"
echo "  uv run compare_models.py --summary results/gcp/summary_merged.json"
echo "  uv run plot_training.py  --results-dir results/gcp --out charts/gcp"
echo "============================================================"
