# HGCLR / RCV1-103-H3 on Runpod: direct A40 Pods with one shared Network Volume

This is the recommended fast path when the same RCV1 preparation must serve
five independent fold Pods. It uses no Runpod template and no custom image.
Create ordinary Pods directly with the standard cached PyTorch image, and mount
one existing Network Volume at `/workspace` in every Pod.

```text
one staging Pod: download RCV1 + prepare once
five A40 fold Pods: each runs one fold concurrently against the prepared data
```

The Network Volume is shared storage, so it removes five repeated downloads,
Conda installations, and preprocessing runs. The Pods remain independent for
GPU work and outputs.

## Execution layout on the shared volume

```text
/workspace/flaviossf/
├── hgclr-source/                  # one pinned public Git checkout
├── miniconda3/                    # one legacy Conda HGCLR environment
├── hgclr-runtime.env              # source/Conda/revision manifest
├── hgclr-shared/
│   ├── RCV1-103-H3/               # download + preprocessing; read-only after READY
│   └── .cache/huggingface/
└── hgclr-runs/
    ├── fold-0/HGCLR/resource/     # output isolated to fold 0
    ├── fold-1/HGCLR/resource/
    ├── fold-2/HGCLR/resource/
    ├── fold-3/HGCLR/resource/
    └── fold-4/HGCLR/resource/
```

`prepare_standard_rcv1.sh` runs exactly once. It writes RCV1 metadata,
`tok.bin`, `slot.pt`, `Y.bin`, split artifacts, and `READY`. Fold Pods only
read these files and write only inside their own `hgclr-runs/fold-N` directory.

## 1. Use one existing Network Volume

Use the same existing volume (for example, the 256 GB volume already created)
for staging and every fold Pod. Do **not** create one volume per fold.

Every Pod that uses it must:

- be in the same Runpod datacenter/region as the volume;
- select that existing volume in the deployment form;
- mount it exactly at `/workspace`.

If the volume is absent from the Pod creation dropdown, the selected
datacenter/region differs from the volume's location.

## 2. Create the staging Pod directly

In **Runpod Console → Deploy**, create an ordinary Pod:

| Field | Value |
|---|---|
| GPU | A40 |
| Container image | `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04` |
| Container disk | 20 GB minimum |
| Network Volume | the one existing shared volume |
| Volume mount path | `/workspace` |
| Container Start Command | `bash -lc 'exec sleep infinity'` |
| Template | none |

Give only the staging Pod the Hugging Face secret as an environment variable:

```text
HF_TOKEN={{ RUNPOD_SECRET_hf_token }}
```

Do not put the token in a command, Git file, or terminal history.

When the Pod is `Running`, verify the A40 and mount:

```bash
nvidia-smi
df -h /workspace
```

## 3. Bootstrap the shared HGCLR runtime once

Use a full immutable 40-character commit SHA. Run this in the staging Pod:

```bash
export HGCLR_REVISION=6f674978c96c02b8347272f41d190116b6a68f60
export WORKSPACE_ROOT=/workspace/flaviossf

curl --fail --location --retry 3 --silent --show-error \
  "https://raw.githubusercontent.com/flaviosoriano/hgclr-rcv1-runpod/${HGCLR_REVISION}/deploy/runpod-hgclr/scripts/bootstrap_standard_runpod.sh" \
  --output /tmp/hgclr-bootstrap.sh

bash /tmp/hgclr-bootstrap.sh
```

This creates Conda, the pinned checkout, and `hgclr-runtime.env` on the shared
volume. It needs to run only once. The A40 is supported: it is Ampere (`sm_86`)
and is compatible with HGCLR's PyTorch 1.8.1 / CUDA 11.1 environment.

## 4. Stage RCV1 once

Still in the staging Pod:

```bash
bash /workspace/flaviossf/hgclr-source/deploy/runpod-hgclr/scripts/prepare_standard_rcv1.sh
```

Wait for this success message:

```text
Shared dataset prepared: /workspace/flaviossf/hgclr-shared/RCV1-103-H3
```

Then verify:

```bash
cat /workspace/flaviossf/hgclr-shared/RCV1-103-H3/READY
```

Expected fields:

```text
dataset=RCV1-103-H3
image_revision=6f674978c96c02b8347272f41d190116b6a68f60
prepared_utc=...
```

Do not launch folds before `READY` exists. After it exists, the shared dataset
is intentionally read-only. Stop the staging Pod if it is no longer needed.

## 5. Create five fold Pods directly

Create five ordinary Pods, one for each `FOLD=0`, `1`, `2`, `3`, and `4`.
Every fold Pod uses:

```text
GPU:                  A40
Container image:      runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
Container disk:       20 GB minimum
Network Volume:       the same existing shared volume
Volume mount path:    /workspace
Start Command:        bash -lc 'exec sleep infinity'
HF_TOKEN:             absent
```

No fold Pod needs bootstrap or a Hugging Face token; both the Conda runtime and
prepared dataset are already present on the mounted volume.

In each fold Pod terminal, run only:

```bash
export FOLD=0
bash /workspace/flaviossf/hgclr-source/deploy/runpod-hgclr/scripts/run_standard_fold.sh
```

Change `FOLD=0` to the Pod's assigned value. The runner validates `READY` and
its source revision, creates a Pod-specific code worktree, and executes:

```text
fit → predict → eval
```

All five fold Pods may run concurrently because they only read the shared
prepared data and their outputs are separated by fold.

## 6. Preserve results

Before ending a fold Pod, inspect the volume-backed output:

```text
/workspace/flaviossf/hgclr-runs/fold-N/HGCLR/resource/
```

Because this is on the Network Volume, the files remain after the Pod stops.
Preserve execution-unique artifacts such as checkpoints, predictions, rankings,
reports, logs, timing files, and immutable run configuration. Do not copy
re-downloadable models, caches, virtual environments, or dependencies unless
there is a specific reason.

## Alternative: no Network Volume

`run_ephemeral_fold.sh` remains available for the deliberately independent
mode: one Pod per fold, no shared storage, each Pod downloads/preprocesses RCV1
itself. That mode is useful only when avoiding the volume is more important
than repeated setup. It requires `HF_TOKEN` in every Pod and loses all files
when a Pod is removed.

## Safeguards

- The bootstrap accepts only a 40-character lowercase Git SHA and verifies the
  detached checkout before writing `hgclr-runtime.env`.
- The official RCV1 topic hierarchy, not label order/cooccurrence, creates the
  label taxonomy and slots.
- RCV1 rankings use `text_<text_idx>`; legacy WOS behavior remains unchanged.
- `READY` binds prepared data to the immutable source revision, preventing a
  fold from silently using incompatible preprocessing.
