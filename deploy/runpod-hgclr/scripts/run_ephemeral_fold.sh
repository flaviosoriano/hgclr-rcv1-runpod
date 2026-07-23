#!/usr/bin/env bash
# Run exactly one RCV1 fold in one Pod-local standard-image workspace.
# No shared Network Volume is used: every Pod downloads and preprocesses its
# own copy before fitting its selected fold. The Pod-local disk may be retained
# independently by Runpod, but no other Pod can read it.
set -euo pipefail

FOLD=${FOLD:?Set FOLD to one of 0,1,2,3,4.}
[[ "$FOLD" =~ ^[0-4]$ ]] || { echo "Invalid FOLD: $FOLD" >&2; exit 2; }

WORKSPACE_ROOT=${WORKSPACE_ROOT:-/workspace/flaviossf}
RUNTIME_ENV=${RUNTIME_ENV:-${WORKSPACE_ROOT}/hgclr-runtime.env}
[[ -f "$RUNTIME_ENV" ]] || {
    echo "Bootstrap manifest is absent: $RUNTIME_ENV" >&2
    exit 1
}
# shellcheck source=/dev/null
source "$RUNTIME_ENV"

DATA_DIR=${DATA_DIR:-${WORKSPACE_ROOT}/hgclr-shared/RCV1-103-H3}
READY_FILE="$DATA_DIR/READY"
export DATA_DIR

if [[ -f "$READY_FILE" ]]; then
    grep -Fx "image_revision=$IMAGE_HGCLR_REVISION" "$READY_FILE" >/dev/null || {
        echo "Local RCV1 READY revision differs from this HGCLR runtime: $READY_FILE" >&2
        exit 1
    }
    printf 'reusing local RCV1 staging, then fold %s.\n' "$FOLD"
else
    : "${HF_TOKEN:?Set HF_TOKEN for this Pod; each Pod-local disk downloads RCV1 independently.}"
    printf 'Pod-local RCV1 staging, then fold %s.\n' "$FOLD"
    bash "$IMAGE_SOURCE/deploy/runpod-hgclr/scripts/prepare_standard_rcv1.sh"
fi

# Dataset download is complete (or READY was reused); do not pass the
# credential to training, prediction, or evaluation subprocesses.
unset HF_TOKEN
exec bash "$IMAGE_SOURCE/deploy/runpod-hgclr/scripts/run_standard_fold.sh"
