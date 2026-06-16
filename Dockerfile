FROM python:3.11-slim-bookworm

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir -e . && \
    python -c "from interview_mapper.mediapipe_models import face_landmarker_model; face_landmarker_model()"

ENV HOST=0.0.0.0
ENV PORT=10000
ENV DATA_DIR=/app/data

RUN mkdir -p /app/data/models

EXPOSE 10000

CMD ["sh", "-c", "interview-mapper serve --host 0.0.0.0 --port ${PORT:-10000} --data-dir ${DATA_DIR:-/app/data} --no-open"]
