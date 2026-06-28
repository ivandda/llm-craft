# Training image for src/sft/train.py on Vertex AI Custom Training (1x NVIDIA L4).
# CUDA runtime base provides the libraries torch/bitsandbytes link against;
# the GPU driver is supplied by the Vertex node.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Install the project venv into the image so `uv run` is not needed at runtime.
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    HF_HOME=/tmp/hf-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# uv manages both the Python toolchain and the dependencies.
COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first (cached layer) using only the lock + manifest.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

# Then add the source and finish installing the project itself.
COPY src ./src
COPY configs ./configs
COPY README.md ./README.md
RUN uv sync --frozen --no-dev

ENV PATH="/opt/venv/bin:${PATH}"

# Vertex overrides command/args from the CustomJob; this is the local default.
ENTRYPOINT ["python", "-m", "src.sft.train"]
