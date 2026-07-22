# syntax=docker/dockerfile:1.7
# Build from the repository root:
# docker build --platform linux/amd64 -f deploy/runpod-hgclr/Dockerfile -t REGISTRY/hgclr-rcv1:VERSION .
#
# The Runpod base image retains its supported Jupyter/SSH startup services.
# HGCLR itself runs only inside the pinned Conda environment below.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

USER root
ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hgclr-cache/huggingface

RUN apt-get update --yes \
 && apt-get install --yes --no-install-recommends \
      ca-certificates curl git git-lfs tini util-linux \
 && rm -rf /var/lib/apt/lists/*

# Keep the bootstrap deterministic enough to detect a corrupted installer.
# The checksum manifest is fetched from the same official Miniconda release.
ARG MINICONDA=Miniconda3-py312_25.1.1-2-Linux-x86_64.sh
RUN curl --fail --location --retry 3 \
      "https://repo.anaconda.com/miniconda/${MINICONDA}" -o /tmp/miniconda.sh \
 && curl --fail --location --retry 3 \
      "https://repo.anaconda.com/miniconda/sha256sum.txt" -o /tmp/miniconda.sha256 \
 && grep " ${MINICONDA}$" /tmp/miniconda.sha256 | sha256sum --check --status - \
 && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
 && rm -f /tmp/miniconda.sh /tmp/miniconda.sha256 \
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
# Preserve Runpod's base-image entrypoint/CMD so SSH and Jupyter are available.
