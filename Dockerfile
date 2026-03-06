FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

ARG GO_VERSION=1.24.0

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV VIRTUAL_ENV=/opt/venv
ENV GOTOOLCHAIN=local
ENV PATH="/usr/local/go/bin:$VIRTUAL_ENV/bin:$PATH"

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    ca-certificates \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" -O /tmp/go.tgz \
    && rm -rf /usr/local/go \
    && tar -C /usr/local -xzf /tmp/go.tgz \
    && rm -f /tmp/go.tgz \
    && go version

RUN python3 -m venv $VIRTUAL_ENV

RUN mkdir -p /workspace

# Install Python dependencies
COPY requirements.txt .
RUN $VIRTUAL_ENV/bin/pip install --no-cache-dir --upgrade pip \
    && $VIRTUAL_ENV/bin/pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
