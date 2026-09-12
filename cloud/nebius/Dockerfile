# Build from the repository root, targeting linux/amd64.
ARG PYTHON_IMAGE=python:3.10.16-slim-bookworm@sha256:f9fd9a142c9e3bc54d906053b756eb7e7e386ee1cf784d82c251cf640c502512
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.7.19@sha256:2dcbc74e60ed6d842122ed538f5267c80e7cde4ff1b6e66a199b89972496f033
FROM ${UV_IMAGE} AS uv
FROM ${PYTHON_IMAGE}
COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsdl2-2.0-0 libsdl2-image-2.0-0 libsdl2-mixer-2.0-0 libsdl2-ttf-2.0-0 \
    libgl1 libglib2.0-0 build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/cleanrl
COPY pyproject.toml uv.lock README.md ./
COPY cleanrl/ cleanrl/
COPY cleanrl_utils/ cleanrl_utils/
RUN uv sync --frozen --no-dev --extra cloud
COPY cloud/nebius/ cloud/nebius/
ARG CLEANRL_REVISION
RUN test -n "$CLEANRL_REVISION" && printf '%s\n' "$CLEANRL_REVISION" > cloud/nebius/source-revision.txt
ENV PATH="/opt/cleanrl/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENTRYPOINT ["/opt/cleanrl/.venv/bin/python", "-m", "cloud.nebius.worker"]
