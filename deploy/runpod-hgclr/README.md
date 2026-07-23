# HGCLR / RCV1-103-H3 on Runpod: one A40 Pod-local Volume Disk per fold

This deployment uses five direct A40 Pods, one per fold. Each Pod has its own
individual Volume Disk mounted at `/workspace`; it is **not** a shared Network
Volume. The Pods may run in parallel, but no Pod can read another Pod's disk.

```text
Pod FOLD=0 + its own /workspace: bootstrap → download → preprocess → fold 0
Pod FOLD=1 + its own /workspace: bootstrap → download → preprocess → fold 1
Pod FOLD=2 + its own /workspace: bootstrap → download → preprocess → fold 2
Pod FOLD=3 + its own /workspace: bootstrap → download → preprocess → fold 3
Pod FOLD=4 + its own /workspace: bootstrap → download → preprocess → fold 4
```

This deliberately repeats the RCV1 download, Conda setup, and preprocessing
in every Pod so fold wall-clock time is minimized through parallel execution.
No Runpod template and no custom image are required.

## What is local versus shared

Each Pod has this layout on its own volume disk:

```text
/workspace/flaviossf/
├── hgclr-source/                 # pinned Git checkout, local to this Pod
├── miniconda3/                   # Conda/runtime, local to this Pod
├── hgclr-runtime.env
├── hgclr-shared/RCV1-103-H3/     # name is local; not shared with other Pods
└── hgclr-runs/fold-N/HGCLR/      # selected fold's outputs
```

`hgclr-shared` is only a legacy-compatible directory name inside one Pod. It
does not imply that another Pod sees its files. The small symlink created by
staging also stays inside the same Pod; it is only how legacy HGCLR finds the
local dataset at `resource/dataset/RCV1-103-H3`.

## 1. Create one Pod directly for one fold

In **Runpod Console → Deploy**, create an ordinary Pod with:

| Field | Value |
|---|---|
| GPU | A40 |
| Container image | `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04` |
| Pod-local Volume Disk | your preferred size, mounted at `/workspace` (100 GB recommended) |
| Network Volume | none |
| Container Start Command | `bash -lc 'exec sleep infinity'` |
| Template | none |

In that Pod's deployment form configure:

```text
FOLD=0
HF_TOKEN={{ RUNPOD_SECRET_hf_token }}
```

Use `FOLD=1`, `2`, `3`, or `4` in the corresponding other Pods. Each Pod needs
its own Hugging Face secret configuration because each downloads RCV1 on its
own disk. Never put the token in commands, scripts, Git, or logs.

## 2. Bootstrap this Pod's runtime

When the Pod is `Running`, connect through SSH or Web Terminal:

```bash
nvidia-smi
df -h /workspace
```

Use the same immutable revision in all five Pods:

```bash
export HGCLR_REVISION=6f674978c96c02b8347272f41d190116b6a68f60
export WORKSPACE_ROOT=/workspace/flaviossf

curl --fail --location --retry 3 --silent --show-error \
  "https://raw.githubusercontent.com/flaviosoriano/hgclr-rcv1-runpod/${HGCLR_REVISION}/deploy/runpod-hgclr/scripts/bootstrap_standard_runpod.sh" \
  --output /tmp/hgclr-bootstrap.sh

bash /tmp/hgclr-bootstrap.sh
```

This is per-Pod work. It installs the legacy HGCLR runtime in that Pod's own
volume disk. A40 is supported: it is Ampere (`sm_86`), compatible with HGCLR's
PyTorch 1.8.1 / CUDA 11.1 runtime.

## 3. Run this Pod's fold end-to-end

Run:

```bash
bash /workspace/flaviossf/hgclr-source/deploy/runpod-hgclr/scripts/run_ephemeral_fold.sh
```

Despite its historical filename, `run_ephemeral_fold.sh` means **no shared
volume**. It works with a Pod-local persistent Volume Disk as well.

The script validates `FOLD=0..4`, requires `HF_TOKEN`, and then in this Pod:

1. Downloads `LBD-UFMG/RCV1-103-H3` to its own `/workspace` disk.
2. Generates official RCV1 taxonomy metadata and `row_to_text_idx.pkl`.
3. Preprocesses its own data and writes a local `READY` marker.
4. Removes `HF_TOKEN` from training/prediction/evaluation subprocesses.
5. Runs `fit,predict,eval` for the selected fold only.

Successful output is under:

```text
/workspace/flaviossf/hgclr-runs/fold-N/HGCLR/resource/
```

## 4. Launch all folds in parallel

Repeat sections 1–3 for five Pods with:

```text
FOLD=0
FOLD=1
FOLD=2
FOLD=3
FOLD=4
```

They can run concurrently because they share no disk, cache, staging lock,
preprocessed data, code directory, or output path. Repeated downloads and
preprocessing are expected by design.

## 5. Preserve results from each independent disk

Each Pod's outputs remain on its own Volume Disk only while that disk remains
attached/retained. Before removing a Pod or deleting its disk, preserve its
execution-unique artifacts:

```text
/workspace/flaviossf/hgclr-runs/fold-N/HGCLR/resource/
/workspace/flaviossf/hgclr-runs/fold-N/HGCLR/resource/time/
```

Typical artifacts to keep: checkpoints, predictions, rankings, result reports,
logs, timing data, and immutable run configuration. Do not rely on one Pod's
volume being accessible from another Pod.

## Safeguards

- Bootstrap accepts only a full 40-character lowercase Git SHA and validates
  the detached checkout before writing `hgclr-runtime.env`.
- The RCV1 label hierarchy comes from the versioned official topic taxonomy,
  never inferred from label order or document cooccurrence.
- RCV1 rankings use `text_<text_idx>`; legacy WOS behavior is unchanged.
- Each local `READY` binds local preprocessing to the source revision used by
  that same Pod.
