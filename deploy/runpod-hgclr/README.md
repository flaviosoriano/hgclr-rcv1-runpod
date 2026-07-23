# Runpod standard-image bootstrap: HGCLR / RCV1-103-H3

This is the recommended deployment path for RCV1. It deliberately uses a
standard, cached Runpod PyTorch image for fast Pod creation, then bootstraps
the pinned legacy HGCLR Conda environment into the persistent Network Volume.

The former custom GHCR image is about 6.84 GiB compressed (with one 6.12 GiB
layer). It remains available as a fallback, but it can spend many minutes in
registry pull/unpack before the Pod starts. The standard-image path moves the
slow work into a logged, idempotent bootstrap after the container is running.

The public repository is used only to fetch the exact source revision. RCV1
and Hugging Face credentials are never committed. A public repository means
all code is visible; this is equivalent to the previous public image exposing
its baked source layers.

## Execution layout

```text
/workspace/flaviossf/
├── hgclr-source/              # pinned public Git checkout
├── miniconda3/                # legacy Conda installation and HGCLR env
├── hgclr-runtime.env          # source/Conda/revision manifest
├── hgclr-shared/
│   ├── RCV1-103-H3/           # download + preprocess, immutable after staging
│   └── .cache/huggingface/
└── hgclr-runs/
    ├── fold-0/HGCLR/          # isolated code + outputs
    ├── fold-1/HGCLR/
    └── fold-4/HGCLR/
```

`prepare_shared_rcv1.sh` writes `tok.bin`, `slot.pt`, `Y.bin`, and split
artifacts. Run it exactly once. Five fold Pods only read this prepared data
and write into their own `hgclr-runs/fold-N` paths.

## 1. Create the standard public Runpod template

In **Runpod Console → Templates → New Template** use:

| Field | Value |
|---|---|
| Name | `hgclr-rcv1-standard-bootstrap` |
| Container image | `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04` |
| Container disk | 20 GB minimum |
| Network volume | 150 GB minimum, mounted at `/workspace` |
| Registry authentication | None |
| Template visibility | Public |
| HTTP/TCP ports | None required; use the Runpod Web Terminal |

Use an Ampere GPU: A40, RTX A5000, RTX A6000, or A100. The pinned HGCLR stack
uses PyTorch 1.8.1 with CUDA 11.1 and validates compute capability `8.x`.

Set the template environment variable `HGCLR_REVISION` to a full immutable
40-character commit SHA from this public repository. Do not use `main` for a
measured run.

Set the **Container Start Command** below, replacing `<COMMIT_SHA>` with that
same value. It creates/reuses the Conda environment on the mounted volume,
locks concurrent setup, validates imports, then keeps the Pod alive.

```bash
bash -lc 'set -euo pipefail; export HGCLR_REVISION=<COMMIT_SHA>; curl --fail --location --retry 3 --silent --show-error "https://raw.githubusercontent.com/flaviosoriano/hgclr-rcv1-runpod/${HGCLR_REVISION}/deploy/runpod-hgclr/scripts/bootstrap_standard_runpod.sh" --output /tmp/hgclr-bootstrap.sh; bash /tmp/hgclr-bootstrap.sh; sleep infinity'
```

The first Pod performs Conda installation and environment creation. This can
take time, but it happens after the standard base container starts and leaves
visible logs. Later Pods reuse `/workspace/flaviossf/miniconda3` and the pinned
source checkout, so bootstrap is fast.

## 2. Stage shared RCV1 data once

Create one temporary staging Pod from the template. Attach the shared volume
in the same datacenter as all fold Pods. Give **only this Pod** the secret:

```text
HF_TOKEN={{ RUNPOD_SECRET_hf_token }}
```

After its bootstrap command has completed, use the Web Terminal:

```bash
bash /workspace/flaviossf/hgclr-source/deploy/runpod-hgclr/scripts/prepare_standard_rcv1.sh
```

Wait for success and inspect:

```bash
cat /workspace/flaviossf/hgclr-shared/RCV1-103-H3/READY
```

It must contain the same immutable revision:

```text
dataset=RCV1-103-H3
image_revision=<COMMIT_SHA>
```

Stop the staging Pod when `READY` exists. Never run staging in parallel.

## 3. Pilot fold 0

Create one new Pod from the same template and attach the same volume. Do not
set `HF_TOKEN`. Set only:

```text
FOLD=0
```

After bootstrap has completed, run:

```bash
nvidia-smi
bash /workspace/flaviossf/hgclr-source/deploy/runpod-hgclr/scripts/run_standard_fold.sh
```

The runner performs `fit,predict,eval`. It must read `READY`, reject a source
revision mismatch, and write only under:

```text
/workspace/flaviossf/hgclr-runs/fold-0/HGCLR/resource/
```

Before launching all folds, retain the pilot terminal log and confirm a
ranking uses keys of the form `text_<text_idx>`.

## 4. Launch the remaining folds

After fold 0 passes, create Pods for `FOLD=1`, `FOLD=2`, `FOLD=3`, and
`FOLD=4`, all using the identical template, `HGCLR_REVISION`, and mounted
Network Volume. No fold Pod receives the Hugging Face secret.

Run the same `run_standard_fold.sh` command in each Pod. Each fold writes only
to its own `hgclr-runs/fold-N` directory.

## Operational safeguards

- The bootstrap requires a 40-character commit SHA and checks the checked-out
  Git revision before generating `hgclr-runtime.env`.
- Bootstrap holds a volume-wide `flock`; simultaneous Pods do not create or
  overwrite the Conda environment concurrently.
- `READY` binds staging data to `IMAGE_HGCLR_REVISION`; fold runs reject data
  prepared from a different source revision.
- WOS files/configuration are untouched. This path applies only to RCV1.
- The HF token is used only by staging; it must never be saved in the volume,
  source checkout, template, logs, or fold Pods.

## Legacy custom image fallback

The previously published custom image remains available for controlled
comparison:

```text
ghcr.io/flaviosoriano/hgclr-rcv1:cb1a398da19ba46a69d4fce5ed95422fe811c0af
```

Do not use it for the first retry while diagnosing long initialization. If the
standard base also remains in `Initializing`, inspect the Runpod Pod events
and pull logs: that points to scheduler, GPU availability, region, disk, or
volume attachment rather than the HGCLR image layers.
