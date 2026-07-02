# Container for the recommendation API.
# The service builds the interaction matrix and trains the chosen ALS config at
# startup, so the image bakes in the Last.fm dataset to be self-contained.
FROM python:3.12-slim

ENV OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for fetching the dataset.
RUN apt-get update && apt-get install -y --no-install-recommends curl unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# App code + config.
COPY api ./api
COPY configs ./configs

# Bake in the dataset (the API reads data/raw at startup).
RUN mkdir -p data/raw \
    && curl -sSL -o /tmp/lastfm.zip \
        https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip \
    && unzip -o -q /tmp/lastfm.zip -d data/raw \
    && rm /tmp/lastfm.zip

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
