#!/usr/bin/env bash
# Run this in each of five Pods, one value of FOLD per Pod: 0, 1, 2, 3, or 4.
# Required environment: FOLD. The pinned source revision is baked into the image.

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

FOLD=${FOLD:?Set FOLD to one of 0,1,2,3,4.}
[[ "$FOLD" =~ ^[0-4]$ ]] || { echo "Invalid FOLD: $FOLD" >&2; exit 2; }
DATA_DIR=${DATA_DIR:-${SHARED_ROOT}/RCV1-103-H3}
RUN_ROOT=${RUN_ROOT:-${WORKSPACE_ROOT}/hgclr-runs/fold-${FOLD}}
REPO_ROOT=${REPO_ROOT:-${RUN_ROOT}/HGCLR}

require_hgclr_environment
materialize_worktree "$REPO_ROOT"
link_shared_data "$REPO_ROOT" "$DATA_DIR"

cd "$REPO_ROOT"
hgclr bash run/RCV1-103-H3.sh "$FOLD" "$FOLD" fit,predict,eval

printf 'Fold %s completed. Results remain under %s/resource/.\n' "$FOLD" "$REPO_ROOT"
