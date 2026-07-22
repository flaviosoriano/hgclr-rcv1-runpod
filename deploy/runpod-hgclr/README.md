# Runpod template: HGCLR / RCV1-103-H3

This directory defines a reproducible **Conda** image and two operational scripts:

- `prepare_shared_rcv1.sh`: one staging Pod downloads and preprocesses RCV1 once;
- `run_fold.sh`: five independent training Pods run folds `0` through `4`.

The image intentionally contains **no RCV1 data and no Hugging Face token**. It contains the exact HGCLR source revision passed at build time, the legacy Conda environment from `environment.yml`, and the public `bert-base-uncased` encoder cache. Consequently, Pods never need GitHub credentials to clone the private repository at runtime.

## Why staging is separate from fold execution

`preprocess` writes shared files such as `tok.bin`, `slot.pt`, `Y.bin`, and fold artifacts. Running it in five Pods against one mounted volume would create write races. Stage once, wait for the `READY` manifest, then make all five Pods read the same prepared dataset while writing checkpoints/rankings/results into distinct per-fold worktrees. This prevents overlapping writes to any shared dataset artifact.

```text
/workspace/flaviossf/
├── hgclr-shared/
│   ├── RCV1-103-H3/       # download + preprocessing; shared, read-only after staging
│   └── .cache/huggingface/
└── hgclr-runs/
    ├── fold-0/HGCLR/      # code + outputs for fold 0
    ├── fold-1/HGCLR/
    ├── ...
    └── fold-4/HGCLR/
```

Runpod documents that a network volume can be attached to multiple Pods, but warns that concurrent writes can corrupt it. This design avoids overlapping writes to shared dataset artifacts during the five fold runs.

## 1. Publish the image

The repository workflow `.github/workflows/publish-ghcr.yml` publishes automatically to the private GitHub Container Registry on every push to `main`. The immutable production reference is:

```text
ghcr.io/flaviosoriano/hgclr-rcv1:<FULL_COMMIT_SHA>
```

Do not use the mutable `:main` tag for a measured fold run. The workflow fails its build if the Conda environment cannot import the pinned HGCLR stack or if the public BERT cache cannot be created.

For a manual build from a clean committed checkout:

```bash
git diff --quiet && test -z "$(git status --porcelain)"
REVISION=$(git rev-parse HEAD)
docker build --platform linux/amd64 \
  --build-arg HGCLR_REVISION="$REVISION" \
  -f deploy/runpod-hgclr/Dockerfile \
  -t REGISTRY_NAMESPACE/hgclr-rcv1:"$REVISION" .

docker push REGISTRY_NAMESPACE/hgclr-rcv1:"$REVISION"
```

The Dockerfile fails its build if the Conda environment cannot import the pinned HGCLR stack or if the public BERT cache cannot be created.

Interactive shells automatically activate `HGCLR`. The image also offers a durable wrapper instead of relying only on a shell alias:

```bash
hgclr python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

## 2. Create the private Runpod template

In **Runpod Console → Templates → New Template** use:

| Field | Value |
|---|---|
| Name | `hgclr-rcv1-conda-ampere` |
| Container image | `ghcr.io/flaviosoriano/hgclr-rcv1:<FULL_COMMIT_SHA>` |
| Container disk | **50 GB** minimum (the image contains Conda plus the legacy stack) |
| HTTP/TCP ports | None required; use the Runpod Web Terminal. Add ports only if you later add a service. |
| Registry authentication | Select the saved private `ghcr.io` registry credential |
| Template visibility | Private |

Use an **Ampere** GPU because the project pins PyTorch `1.8.1` with CUDA `11.1`; the preflight deliberately accepts capability `8.x` only. Suitable Pod choices are A40, RTX A5000, RTX A6000, or A100. Start with 24 GB only after a pilot fold; choose 48 GB if the pilot reports CUDA OOM.

Before saving the template, create a GitHub personal access token with only `read:packages` for pulling this private GHCR image. In Runpod, add a private registry authentication for `ghcr.io` with username `flaviosoriano` and that token as its password, then select that credential in the template. Do not place this registry token in an environment variable or in the repository.

### Optional: make the image available to a public Runpod template

GitHub Packages does not provide a REST/GraphQL mutation for changing container-package visibility. In GitHub's web UI, visit:

```text
https://github.com/users/flaviosoriano/packages/container/package/hgclr-rcv1
```

Open **Package settings → Change visibility → Public**, type the package name to confirm, and save. The GitHub repository remains private, but the image layers (and therefore the HGCLR code embedded in them) become publicly readable. Once public, create a public Runpod template with the immutable image tag and **no registry credential**:

```text
ghcr.io/flaviosoriano/hgclr-rcv1:cb1a398da19ba46a69d4fce5ed95422fe811c0af
```

Create an encrypted Runpod secret named, for example, `hf_token`. Runpod references it in a template/environment field as:

```text
{{ RUNPOD_SECRET_hf_token }}
```

Assign that value to the environment variable `HF_TOKEN` **only in the staging Pod**. Never bake or commit the token.

## 3. Stage the shared data once

Create or attach a network volume of at least **100 GB** in the same data center/region as the planned five training Pods, mounted at `/workspace`. Launch one temporary staging Pod from this template and set only:

```text
HF_TOKEN={{ RUNPOD_SECRET_hf_token }}
```

Then run:

```bash
/opt/hgclr/bin/prepare_shared_rcv1.sh
```

Do not launch the five fold Pods until this exits successfully and this exists:

```text
/workspace/flaviossf/hgclr-shared/RCV1-103-H3/READY
```

## 4. Launch the five fold Pods

Attach the same prepared network volume to each Pod. For each Pod set:

```text
FOLD=<0|1|2|3|4>
```

No Hugging Face secret is needed in these Pods. In the Pod terminal run:

```bash
/opt/hgclr/bin/run_fold.sh
```

The fold runner performs `fit,predict,eval`; it does **not** download or preprocess the dataset. Outputs are isolated in:

```text
/workspace/flaviossf/hgclr-runs/fold-FOLD/HGCLR/resource/
```

## Pilot gate before five-Pod launch

Before paying for all five GPUs, launch only `FOLD=0` and record:

1. the immutable image tag and its baked source revision (`IMAGE_HGCLR_REVISION`);
2. `nvidia-smi` and the preflight output (PyTorch/CUDA/GPU capability);
3. successful reading of `READY` and completion of one training epoch;
4. a ranking whose keys use `text_<text_idx>`;
5. checkpoint, ranking, and result locations.

Only after this pilot passes should Pods for folds 1–4 be launched in parallel.
