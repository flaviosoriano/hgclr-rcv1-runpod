# HGCLR / RCV1-103-H3 on Runpod: one ephemeral A40 Pod per fold

This deployment intentionally uses **no Network Volume**, no template, and no
custom image. Create five ordinary A40 Pods directly in Runpod. Each Pod
downloads and preprocesses its own RCV1 copy, then runs exactly one fold.

```text
Pod FOLD=0: bootstrap → download RCV1 → preprocess → fit → predict → eval
Pod FOLD=1: bootstrap → download RCV1 → preprocess → fit → predict → eval
Pod FOLD=2: bootstrap → download RCV1 → preprocess → fit → predict → eval
Pod FOLD=3: bootstrap → download RCV1 → preprocess → fit → predict → eval
Pod FOLD=4: bootstrap → download RCV1 → preprocess → fit → predict → eval
```

This maximizes fold-level parallelism at the cost of five independent dataset
downloads, Conda installations, and preprocessing runs. It removes shared
volume setup and shared-write coordination entirely.

## Important properties

- Use the standard Runpod image, not the legacy GHCR image.
- The A40 is supported: it is Ampere (`sm_86`), compatible with HGCLR's
  PyTorch 1.8.1 / CUDA 11.1 runtime.
- Each Pod needs `HF_TOKEN` only to download its own RCV1 copy.
- A Pod's data, environment, checkpoints, rankings, logs, and results are on
  its ephemeral container disk. Export each fold's results before stopping or
  removing that Pod.
- No Pod reads data from another Pod. `READY` is local to each Pod and only
  guards that Pod's own preprocessing → fold sequence.

## 1. Create one Pod directly for one fold

In **Runpod Console → Deploy**, create an ordinary Pod with:

| Field | Value |
|---|---|
| GPU | A40 |
| Container image | `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04` |
| Container disk | 100 GB recommended |
| Network Volume | **None** |
| Container Start Command | `bash -lc 'exec sleep infinity'` |
| Template | **None** |

Set this Pod's environment variables in the deployment form:

```text
FOLD=0
HF_TOKEN={{ RUNPOD_SECRET_hf_token }}
```

Use a different `FOLD` value for each Pod: `0`, `1`, `2`, `3`, or `4`.
Do not place the token in a command, script, log, or Git file. Configure it
through the Runpod secret/environment-variable UI for every Pod.

Wait for the Pod to be `Running`, then connect through SSH or the Web Terminal.
Confirm the A40 is visible:

```bash
nvidia-smi
```

## 2. Bootstrap the pinned runtime in that Pod

Choose a full immutable 40-character Git commit SHA, then use the same SHA in
all five Pods. From the Pod terminal:

```bash
export HGCLR_REVISION=<COMMIT_SHA>
export WORKSPACE_ROOT=/workspace/flaviossf

curl --fail --location --retry 3 --silent --show-error \
  "https://raw.githubusercontent.com/flaviosoriano/hgclr-rcv1-runpod/${HGCLR_REVISION}/deploy/runpod-hgclr/scripts/bootstrap_standard_runpod.sh" \
  --output /tmp/hgclr-bootstrap.sh

bash /tmp/hgclr-bootstrap.sh
```

The bootstrap creates this Pod-local layout:

```text
/workspace/flaviossf/
├── hgclr-source/                 # exact detached Git checkout
├── miniconda3/                   # legacy HGCLR Conda environment
├── hgclr-runtime.env             # revision/runtime manifest
├── hgclr-shared/RCV1-103-H3/     # this Pod's own download + preprocessing
└── hgclr-runs/fold-N/HGCLR/      # this Pod's own outputs
```

## 3. Run this Pod's local staging and fold

After bootstrap succeeds, run:

```bash
bash /workspace/flaviossf/hgclr-source/deploy/runpod-hgclr/scripts/run_ephemeral_fold.sh
```

The runner validates `FOLD=0..4`, requires `HF_TOKEN`, then does all of the
following in the current Pod only:

1. Downloads `LBD-UFMG/RCV1-103-H3` into its local container disk.
2. Builds the official RCV1 taxonomy metadata and `row_to_text_idx` mapping.
3. Preprocesses its local dataset and writes the local `READY` marker.
4. Removes `HF_TOKEN` from the training subprocess environment.
5. Runs `fit,predict,eval` only for the selected fold.

A successful fold prints a path like:

```text
Fold 0 completed. Results remain under /workspace/flaviossf/hgclr-runs/fold-0/HGCLR/resource/.
```

Do not run a second fold in the same Pod. The one-Pod-per-fold constraint keeps
outputs and failure recovery independent.

## 4. Launch folds in parallel

Repeat sections 1–3 for five Pods with:

```text
FOLD=0
FOLD=1
FOLD=2
FOLD=3
FOLD=4
```

They may run concurrently because they do not share a filesystem, lock, cache,
preprocessed dataset, or output directory. Each needs its own `HF_TOKEN` in
its Pod deployment configuration.

## 5. Export results before ending a Pod

Before stopping/removing each Pod, inspect and export the execution-unique
artifacts from:

```text
/workspace/flaviossf/hgclr-runs/fold-N/HGCLR/resource/
/workspace/flaviossf/hgclr-runs/fold-N/HGCLR/resource/time/
```

Typical artifacts to preserve are checkpoints, predictions, rankings, result
reports, logs, timing files, and the immutable revision/configuration used for
the run. Do not rely on the container disk surviving Pod removal.

## Safeguards

- The bootstrap accepts only a 40-character lowercase Git SHA and checks the
  detached checkout before writing `hgclr-runtime.env`.
- The RCV1 taxonomy comes from the versioned official code hierarchy, not label
  order or label cooccurrence in documents.
- RCV1 rankings use `text_<text_idx>`; legacy WOS behavior is unchanged.
- The folder named `hgclr-shared` is only local naming inside one Pod in this
  mode. It is not a Runpod Network Volume and is never shared with another Pod.
