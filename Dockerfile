# DGX Spark Model Manager - Dockerfile
# Multi-stage build for smaller final image

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Final stage
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    curl \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Ensure Docker Python client works with Unix socket
ENV DOCKER_HOST=unix:///var/run/docker.sock

WORKDIR /app

COPY app/ ./app/
COPY config/ ./config/
COPY engines/ ./engines/

RUN mkdir -p /models

# Add app user to docker group for socket access
RUN groupadd -f docker && usermod -aG docker root

# Ensure Docker socket is accessible
RUN chmod 666 /var/run/docker.sock 2>/dev/null || true

EXPOSE 4600

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:4600/api/status || exit 1

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "4600"]