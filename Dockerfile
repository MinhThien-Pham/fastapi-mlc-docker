FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ARG GO_VERSION=1.24.0
ARG CONDA_VERSION=latest

# ── Environment variables ─────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CONDA_DIR=/opt/conda
ENV CLI_VENV=mlc-cli-venv
ENV GOTOOLCHAIN=local

# Runtime environment variables for baked mlc-cli workspace
ENV MLC_CLI_PATH=/workspace/mlc-cli
ENV BAKED_MLC_CLI_PATH=/opt/mlc-cli

# conda first so conda-managed python/cmake/rust take precedence
ENV PATH="${CONDA_DIR}/bin:/usr/local/go/bin:${PATH}"

# Build config — overridable at runtime via docker-compose / docker run -e
ENV BUILD_ACTION=full
ENV CUDA_ARCH=86
ENV TVM_SOURCE=bundled
ENV BUILD_WHEELS=y
ENV MLC_DEVICE=cuda

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    curl \
    ca-certificates \
    build-essential \
    git \
    git-lfs \
    rsync \
    libxml2-dev \
    zlib1g-dev \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# ── Go ────────────────────────────────────────────────────────────────────────
RUN wget -q "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" -O /tmp/go.tgz \
    && rm -rf /usr/local/go \
    && tar -C /usr/local -xzf /tmp/go.tgz \
    && rm -f /tmp/go.tgz \
    && go version

# ── Miniconda ─────────────────────────────────────────────────────────────────
RUN wget -q "https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh" \
    -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda clean -afy \
    && conda --version

# Initialise conda for non-interactive bash shells used by subprocess.run
RUN conda init bash \
    && echo "conda activate base" >> /root/.bashrc

# ── Baked mlc-cli source ──────────────────────────────────────────────────────
COPY docker/mlc-cli.lock /tmp/mlc-cli.lock
RUN . /tmp/mlc-cli.lock \
    && git clone "${MLC_CLI_REPO}" /opt/mlc-cli \
    && git -C /opt/mlc-cli checkout "${MLC_CLI_REF}" \
    && git -C /opt/mlc-cli rev-parse HEAD > /opt/mlc-cli-ref.txt \
    && echo "${MLC_CLI_REPO}" > /opt/mlc-cli-repo.txt \
    && test -f /opt/mlc-cli/scripts/config/versions.sh

# ── Python venv for FastAPI + mlc-cli ─────────────────────────────────────────
# Create the conda environment from mlc-cli's version source of truth.
RUN bash -lc '\
    set -euo pipefail; \
    source /opt/mlc-cli/scripts/config/versions.sh; \
    conda create -y -n "${CLI_VENV}" -c "${CONDA_CHANNEL}" \
    "python=${PYTHON_VERSION}" \
    "cmake>=${CMAKE_MIN_VERSION}" \
    rust \
    psutil \
    transformers \
    pip; \
    conda clean -afy; \
    conda run -n ${CLI_VENV} python --version \
'
# ── Workspace volume for mlc-cli runtime artifacts ────────────────────────────
RUN mkdir -p /workspace

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN conda run -n ${CLI_VENV} pip install --no-cache-dir --upgrade pip \
    && conda run -n ${CLI_VENV} pip install --no-cache-dir -r requirements.txt

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY docker/entrypoint.sh /usr/local/bin/fastapi-mlc-entrypoint
RUN chmod +x /usr/local/bin/fastapi-mlc-entrypoint

# ── App source ────────────────────────────────────────────────────────────────
COPY . .

EXPOSE 8000

CMD ["/usr/local/bin/fastapi-mlc-entrypoint"]
