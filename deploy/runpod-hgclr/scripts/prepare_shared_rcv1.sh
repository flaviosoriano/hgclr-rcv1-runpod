#!/usr/bin/env bash
# Run ONCE in a staging Pod before launching the five independent fold Pods.
# Required environment: HF_TOKEN (set from a Runpod secret).

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

: "${HF_TOKEN:?Set HF_TOKEN from a Runpod secret; never place it in this script.}"
DATA_DIR=${DATA_DIR:-${SHARED_ROOT}/RCV1-103-H3}
STAGING_REPO=${STAGING_REPO:-${SHARED_ROOT}/staging/HGCLR}

mkdir -p "$SHARED_ROOT"
exec 9>"${SHARED_ROOT}/.rcv1-prepare.lock"
flock -n 9 || { echo 'Another RCV1 staging process holds the preparation lock.' >&2; exit 1; }

require_hgclr_environment
materialize_worktree "$STAGING_REPO"
mkdir -p "$DATA_DIR"

# Persist the Hub cache on the volume; snapshot_download resumes interrupted transfers.
export HF_HOME="${SHARED_ROOT}/.cache/huggingface"
export DATA_DIR
hgclr python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='LBD-UFMG/RCV1-103-H3',
    repo_type='dataset',
    local_dir=os.environ['DATA_DIR'],
    token=os.environ['HF_TOKEN'],
)
PY

cd "$STAGING_REPO"
hgclr python source/helper/rcv1_preparation.py --data-dir "$DATA_DIR"

# This creates global preprocessing files and all fold split artifacts exactly
# once.  Do not run preprocessing simultaneously from the five fold Pods.
hgclr bash run/RCV1-103-H3.sh 0 4 preprocess

for file in samples.pkl label_taxonomy.pkl row_to_text_idx.pkl tok.bin slot.pt; do
    [[ -f "$DATA_DIR/$file" ]] || { echo "Missing expected artifact: $file" >&2; exit 1; }
done

python3 - <<PY
from datetime import datetime, timezone
from pathlib import Path
p = Path(${DATA_DIR@Q}) / 'READY'
p.write_text(
    'dataset=RCV1-103-H3\n'
    'image_revision=' + ${IMAGE_HGCLR_REVISION@Q} + '\n'
    'prepared_utc=' + datetime.now(timezone.utc).isoformat() + '\n'
)
PY

# Training Pods only read these shared artifacts.  Separate fold outputs remain
# in their own worktrees, preventing cross-fold write races.
chmod -R a-w "$DATA_DIR"
printf 'Shared dataset prepared: %s\n' "$DATA_DIR"
