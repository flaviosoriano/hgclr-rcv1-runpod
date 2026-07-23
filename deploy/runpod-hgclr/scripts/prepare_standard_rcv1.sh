#!/usr/bin/env bash
# Execute once after bootstrap_standard_runpod.sh on the staging Pod.
set -euo pipefail

WORKSPACE_ROOT=${WORKSPACE_ROOT:-/workspace/flaviossf}
RUNTIME_ENV=${RUNTIME_ENV:-${WORKSPACE_ROOT}/hgclr-runtime.env}
[[ -f "$RUNTIME_ENV" ]] || {
    echo "Bootstrap manifest is absent: $RUNTIME_ENV" >&2
    exit 1
}
# shellcheck source=/dev/null
source "$RUNTIME_ENV"

exec bash "$IMAGE_SOURCE/deploy/runpod-hgclr/scripts/prepare_shared_rcv1.sh"
