#!/usr/bin/env bash
# Bootstrap HGCLR on a standard Runpod PyTorch image. Run once on the shared
# Network Volume before staging RCV1. It clones the public deployment repo at
# an immutable commit and creates the legacy Conda environment on the volume.

set -euo pipefail

: "${HGCLR_REVISION:?Set HGCLR_REVISION to the 40-character commit SHA to run.}"
[[ "$HGCLR_REVISION" =~ ^[0-9a-f]{40}$ ]] || {
    echo 'HGCLR_REVISION must be a 40-character lowercase Git commit SHA.' >&2
    exit 2
}

WORKSPACE_ROOT=${WORKSPACE_ROOT:-/workspace/flaviossf}
CONDA_DIR=${CONDA_DIR:-${WORKSPACE_ROOT}/miniconda3}
HGCLR_ENV_NAME=${HGCLR_ENV_NAME:-HGCLR}
IMAGE_SOURCE=${IMAGE_SOURCE:-${WORKSPACE_ROOT}/hgclr-source}
RUNTIME_ENV=${RUNTIME_ENV:-${WORKSPACE_ROOT}/hgclr-runtime.env}
REPO_URL=${REPO_URL:-https://github.com/flaviosoriano/hgclr-rcv1-runpod.git}
MINICONDA=Miniconda3-py312_25.1.1-2-Linux-x86_64.sh
MINICONDA_SHA256=4766d85b5f7d235ce250e998ebb5a8a8210cbd4f2b0fea4d2177b3ed9ea87884
LOCK_FILE=${WORKSPACE_ROOT}/.hgclr-bootstrap.lock

mkdir -p "$WORKSPACE_ROOT"
exec 9>"$LOCK_FILE"
flock 9

if command -v apt-get >/dev/null 2>&1; then
    [[ "$(id -u)" == 0 ]] || {
        echo 'Standard-image bootstrap needs root to install git/curl/build tools.' >&2
        exit 1
    }
    apt-get update --yes
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        bzip2 build-essential ca-certificates curl git util-linux
    rm -rf /var/lib/apt/lists/*
fi

if [[ -d "$IMAGE_SOURCE/.git" ]]; then
    git -C "$IMAGE_SOURCE" fetch --depth 1 origin "$HGCLR_REVISION"
else
    rm -rf "$IMAGE_SOURCE"
    git clone --no-checkout "$REPO_URL" "$IMAGE_SOURCE"
    git -C "$IMAGE_SOURCE" fetch --depth 1 origin "$HGCLR_REVISION"
fi
git -C "$IMAGE_SOURCE" checkout --detach --force FETCH_HEAD
[[ "$(git -C "$IMAGE_SOURCE" rev-parse HEAD)" == "$HGCLR_REVISION" ]] || {
    echo 'Checked-out source revision differs from HGCLR_REVISION.' >&2
    exit 1
}
printf '%s\n' "$HGCLR_REVISION" > "$IMAGE_SOURCE/IMAGE_HGCLR_REVISION"

if [[ ! -x "$CONDA_DIR/bin/conda" ]]; then
    # The official installer refuses paths without a .sh suffix, even when
    # invoked with bash. Keep the suffix while still using a safe temp file.
    installer=$(mktemp --suffix=.sh /tmp/hgclr-miniconda-XXXXXX)
    trap 'rm -f "$installer"' EXIT
    curl --fail --location --retry 3 \
        "https://repo.anaconda.com/miniconda/${MINICONDA}" --output "$installer"
    echo "${MINICONDA_SHA256}  ${installer}" | sha256sum --check --status -
    bash "$installer" -b -p "$CONDA_DIR"
    "$CONDA_DIR/bin/conda" config --system --set channel_priority flexible
fi

if ! "$CONDA_DIR/bin/conda" env list | awk '{print $1}' | grep -Fx "$HGCLR_ENV_NAME" >/dev/null; then
    "$CONDA_DIR/bin/conda" env create --file "$IMAGE_SOURCE/environment.yml"
    "$CONDA_DIR/bin/conda" run --no-capture-output --name "$HGCLR_ENV_NAME" \
        pip install --no-cache-dir huggingface_hub==0.24.7
fi

"$CONDA_DIR/bin/conda" run --no-capture-output --name "$HGCLR_ENV_NAME" python - <<'PY'
import fairseq, hydra, torch, transformers
import torch_geometric, torch_scatter, torch_sparse
assert torch.__version__.startswith('1.8.1'), torch.__version__
assert torch.version.cuda == '11.1', torch.version.cuda
print('HGCLR bootstrap environment OK:', torch.__version__, torch.version.cuda)
PY

cat > "$RUNTIME_ENV" <<EOF
export WORKSPACE_ROOT=$(printf '%q' "$WORKSPACE_ROOT")
export CONDA_DIR=$(printf '%q' "$CONDA_DIR")
export HGCLR_ENV_NAME=$(printf '%q' "$HGCLR_ENV_NAME")
export IMAGE_SOURCE=$(printf '%q' "$IMAGE_SOURCE")
export IMAGE_HGCLR_REVISION=$(printf '%q' "$HGCLR_REVISION")
EOF
chmod 0644 "$RUNTIME_ENV"
printf 'Bootstrap complete. Source=%s revision=%s Conda=%s\n' \
    "$IMAGE_SOURCE" "$HGCLR_REVISION" "$CONDA_DIR"
