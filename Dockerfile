FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
RUN pip install --no-cache-dir -e ".[torch]"

ENV CAMERA_GEOMETRY_HOME=/data/cache
RUN mkdir -p /data/cache
VOLUME ["/data"]

EXPOSE 8300
CMD ["uvicorn", "camera_geometry.service:app", "--host", "0.0.0.0", "--port", "8300"]
