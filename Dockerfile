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
# conda first so conda-managed python/cmake/rust take precedence
ENV PATH="${CONDA_DIR}/bin:/usr/local/go/bin:${PATH}"

# Runtime environment variables for MLC/TVM inside FastAPI
ENV TVM_HOME=/workspace/mlc-cli/tvm
ENV PYTHONPATH=/workspace/mlc-cli/tvm/python:${PYTHONPATH:-}
ENV LD_LIBRARY_PATH=/workspace/mlc-cli/tvm/build/lib:/workspace/mlc-cli/mlc-llm/build/lib:${LD_LIBRARY_PATH:-}

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

# ── Python venv for FastAPI app ───────────────────────────────────────────────
# Create the conda environment that mlc-cli expects, so FastAPI runs in it too.
RUN conda create -y -n ${CLI_VENV} python=3.13 pip cmake rust psutil transformers

# ── Workspace (mlc-cli repo lives here via Docker volume) ─────────────────────
RUN mkdir -p /workspace

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN conda run -n ${CLI_VENV} pip install --no-cache-dir --upgrade pip \
    && conda run -n ${CLI_VENV} pip install --no-cache-dir -r requirements.txt

# ── App source ────────────────────────────────────────────────────────────────
COPY . .

EXPOSE 8000

CMD ["conda", "run", "--no-capture-output", "-n", "mlc-cli-venv", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
