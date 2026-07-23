#!/usr/bin/env bash
# Execute in one standard-image Pod per fold after the shared staging succeeds.
set -euo pipefail

: "${FOLD:?Set FOLD to one of 0,1,2,3,4.}"
[[ "$FOLD" =~ ^[0-4]$ ]] || { echo "Invalid FOLD: $FOLD" >&2; exit 2; }

WORKSPACE_ROOT=${WORKSPACE_ROOT:-/workspace/flaviossf}
RUNTIME_ENV=${RUNTIME_ENV:-${WORKSPACE_ROOT}/hgclr-runtime.env}
[[ -f "$RUNTIME_ENV" ]] || {
    echo "Bootstrap manifest is absent: $RUNTIME_ENV" >&2
    exit 1
}
# shellcheck source=/dev/null
source "$RUNTIME_ENV"

exec bash "$IMAGE_SOURCE/deploy/runpod-hgclr/scripts/run_fold.sh"
