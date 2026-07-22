# syntax=docker/dockerfile:1.7
# Build from the repository root:
# docker build --platform linux/amd64 -f deploy/runpod-hgclr/Dockerfile -t REGISTRY/hgclr-rcv1:VERSION .
#
# A compact NVIDIA CUDA base keeps the GitHub Actions build below the runner's
# disk limit. Conda supplies the project-pinned PyTorch/CUDA user-space stack.
FROM nvidia/cuda:11.1.1-base-ubuntu20.04

USER root
ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hgclr-cache/huggingface

RUN apt-get update --yes \
 && apt-get install --yes --no-install-recommends \
      bzip2 build-essential ca-certificates curl git tini util-linux \
 && rm -rf /var/lib/apt/lists/*

# Pin and verify the official Miniconda installer used to create the legacy
# Python 3.8 HGCLR environment.
ARG MINICONDA=Miniconda3-py312_25.1.1-2-Linux-x86_64.sh
ARG MINICONDA_SHA256=4766d85b5f7d235ce250e998ebb5a8a8210cbd4f2b0fea4d2177b3ed9ea87884
RUN curl --fail --location --retry 3 \
      "https://repo.anaconda.com/miniconda/${MINICONDA}" -o /tmp/miniconda.sh \
 && echo "${MINICONDA_SHA256}  /tmp/miniconda.sh" | sha256sum --check --status - \
 && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
 && rm -f /tmp/miniconda.sh \
 && conda config --system --set channel_priority flexible \
 && conda clean --all --yes

COPY environment.yml /tmp/hgclr-environment.yml
RUN conda env create --file /tmp/hgclr-environment.yml \
 && conda run --no-capture-output --name HGCLR pip install --no-cache-dir huggingface_hub==0.24.7 \
 && conda run --no-capture-output --name HGCLR python - <<'PY'
import fairseq, hydra, torch, transformers
import torch_geometric, torch_scatter, torch_sparse
assert torch.__version__.startswith('1.8.1'), torch.__version__
assert torch.version.cuda == '11.1', torch.version.cuda
print('HGCLR Conda environment OK:', torch.__version__, torch.version.cuda)
PY

# Cache only the public base encoder.  The private RCV1 dataset is intentionally
# absent from the image and is fetched into a network volume with a Runpod secret.
RUN conda run --no-capture-output --name HGCLR python - <<'PY'
from transformers import AutoModel, AutoTokenizer
AutoTokenizer.from_pretrained('bert-base-uncased')
AutoModel.from_pretrained('bert-base-uncased')
print('bert-base-uncased cache OK')
PY

# Bake the exact project revision into the image.  Pods never need credentials
# to clone this private repository at runtime.
ARG HGCLR_REVISION=UNSET
RUN test "$HGCLR_REVISION" != "UNSET"
ENV IMAGE_HGCLR_REVISION=${HGCLR_REVISION}
COPY . /opt/hgclr/source/
RUN test -f /opt/hgclr/source/main.py \
 && test -f /opt/hgclr/source/run/RCV1-103-H3.sh \
 && printf '%s\n' "$IMAGE_HGCLR_REVISION" > /opt/hgclr/source/IMAGE_HGCLR_REVISION

COPY deploy/runpod-hgclr/scripts/ /opt/hgclr/bin/
RUN chmod 0755 /opt/hgclr/bin/*.sh \
 && printf '%s\n' \
      '#!/usr/bin/env bash' \
      'exec /opt/conda/bin/conda run --no-capture-output --name HGCLR "$@"' \
      > /usr/local/bin/hgclr \
 && chmod 0755 /usr/local/bin/hgclr \
 && printf '%s\n' \
      'source /opt/conda/etc/profile.d/conda.sh' \
      'conda activate HGCLR' \
      "alias hgclr='/usr/local/bin/hgclr'" \
      > /etc/profile.d/hgclr-conda.sh

WORKDIR /workspace/flaviossf
# Keep the Pod alive for the Runpod web terminal.  Training is explicitly
# started with /opt/hgclr/bin/prepare_shared_rcv1.sh or run_fold.sh.
CMD ["bash", "-lc", "sleep infinity"]
