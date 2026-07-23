#!/usr/bin/env bash
# Run exactly one RCV1 fold in one Pod-local standard-image workspace.
# No shared Network Volume is used: every Pod downloads and preprocesses its
# own copy before fitting its selected fold. The Pod-local disk may be retained
# independently by Runpod, but no other Pod can read it.
set -euo pipefail

FOLD=${FOLD:?Set FOLD to one of 0,1,2,3,4.}
[[ "$FOLD" =~ ^[0-4]$ ]] || { echo "Invalid FOLD: $FOLD" >&2; exit 2; }
: "${HF_TOKEN:?Set HF_TOKEN for this Pod; each ephemeral Pod downloads RCV1 independently.}"

WORKSPACE_ROOT=${WORKSPACE_ROOT:-/workspace/flaviossf}
RUNTIME_ENV=${RUNTIME_ENV:-${WORKSPACE_ROOT}/hgclr-runtime.env}
[[ -f "$RUNTIME_ENV" ]] || {
    echo "Bootstrap manifest is absent: $RUNTIME_ENV" >&2
    exit 1
}
# shellcheck source=/dev/null
source "$RUNTIME_ENV"

printf 'Ephemeral RCV1 mode: Pod-local staging, then fold %s.\n' "$FOLD"
bash "$IMAGE_SOURCE/deploy/runpod-hgclr/scripts/prepare_standard_rcv1.sh"

# Dataset download is complete; do not pass the credential to training,
# prediction, or evaluation subprocesses.
unset HF_TOKEN
exec bash "$IMAGE_SOURCE/deploy/runpod-hgclr/scripts/run_standard_fold.sh"
