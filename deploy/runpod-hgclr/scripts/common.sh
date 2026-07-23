#!/usr/bin/env bash
# Shared helpers for the staging and fold scripts. This file is sourced.

set -euo pipefail

WORKSPACE_ROOT=${WORKSPACE_ROOT:-/workspace/flaviossf}
SHARED_ROOT=${SHARED_ROOT:-${WORKSPACE_ROOT}/hgclr-shared}
IMAGE_HGCLR_REVISION=${IMAGE_HGCLR_REVISION:?Image is missing its baked HGCLR revision.}
IMAGE_SOURCE=${IMAGE_SOURCE:-/opt/hgclr/source}
CONDA_DIR=${CONDA_DIR:-/opt/conda}
HGCLR_ENV_NAME=${HGCLR_ENV_NAME:-HGCLR}

# Works for both the baked custom image and the standard Runpod image after
# bootstrap_standard_runpod.sh has created Conda on the persistent volume.
hgclr() {
    [[ -x "$CONDA_DIR/bin/conda" ]] || {
        echo "Missing Conda executable: $CONDA_DIR/bin/conda" >&2
        return 1
    }
    "$CONDA_DIR/bin/conda" run --no-capture-output --name "$HGCLR_ENV_NAME" "$@"
}

require_hgclr_environment() {
    hgclr python - <<'PY'
import fairseq, hydra, torch, transformers
assert torch.cuda.is_available(), 'No CUDA GPU is visible in this Pod.'
cc = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
assert cc[0] == 8, (
    f'{name} has compute capability {cc}; this PyTorch 1.8.1/CUDA 11.1 image is '
    'validated for Ampere GPUs (A40, RTX A5000/A6000, A100).'
)
print({'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
       'gpu': name, 'compute_capability': cc,
       'transformers': transformers.__version__, 'fairseq': fairseq.__version__})
PY
}

materialize_worktree() {
    local target=$1
    local manifest="$target/IMAGE_HGCLR_REVISION"
    [[ -d "$IMAGE_SOURCE" ]] || { echo "Missing bundled project source: $IMAGE_SOURCE" >&2; return 1; }

    if [[ -e "$target" ]]; then
        [[ -f "$manifest" ]] || { echo "Existing worktree lacks a revision manifest: $target" >&2; return 1; }
        [[ "$(<"$manifest")" == "$IMAGE_HGCLR_REVISION" ]] || {
            echo "Existing worktree was made from another image revision: $target" >&2
            return 1
        }
        return 0
    fi

    mkdir -p "$(dirname "$target")"
    cp -a "$IMAGE_SOURCE" "$target"
    [[ "$(<"$manifest")" == "$IMAGE_HGCLR_REVISION" ]] || {
        echo 'Bundled source revision manifest is inconsistent.' >&2
        return 1
    }
}

link_shared_data() {
    local repo_root=$1
    local data_dir=$2
    local link_path="$repo_root/resource/dataset/RCV1-103-H3"
    local ready="$data_dir/READY"
    [[ -f "$ready" ]] || {
        echo "Shared RCV1 data is not ready: $ready is absent." >&2
        return 1
    }
    grep -Fx "image_revision=$IMAGE_HGCLR_REVISION" "$ready" >/dev/null || {
        echo "Prepared data was created for a different source revision: $ready" >&2
        return 1
    }
    mkdir -p "$(dirname "$link_path")"
    if [[ -L "$link_path" ]]; then
        [[ "$(readlink -f "$link_path")" == "$(readlink -f "$data_dir")" ]] || {
            echo "Existing data symlink points elsewhere: $link_path" >&2
            return 1
        }
    elif [[ -e "$link_path" ]]; then
        echo "Refusing to replace non-symlink data path: $link_path" >&2
        return 1
    else
        ln -s "$data_dir" "$link_path"
    fi
}
