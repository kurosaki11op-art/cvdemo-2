# CPU image. For an NVIDIA GPU use a CUDA base image and drop the cpu-only torch line.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 YOLO_CONFIG_DIR=/tmp/ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt
COPY nexus ./nexus
RUN pip install --no-deps .

# Model weights download on first start into the /data volume (persisted). For offline or
# air-gapped sites, put your own weights in the volume and set `detection.model` to its path.

RUN useradd --create-home nexus && mkdir -p /data && chown nexus /data
USER nexus
WORKDIR /data
VOLUME /data

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD nexus healthcheck --config /config/nexus.yaml || exit 1

ENTRYPOINT ["nexus"]
CMD ["run", "--config", "/config/nexus.yaml"]
