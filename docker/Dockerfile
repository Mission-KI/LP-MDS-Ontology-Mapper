# First Stage: Build environment
FROM python:3.12-slim AS build

# Install git and g++ for building libsvm
RUN apt-get update && \
    apt-get install -y git build-essential

ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

# Set up a virtual environment and install dependencies from pyproject.toml
ARG WHEELS_DIR="/wheels"
RUN --mount=type=secret,id=REPO_TOKEN,required=true \
    --mount=type=cache,mode=0755,target=/root/.cache/pip \
    --mount=type=bind,src=dist,target=${WHEELS_DIR},readonly \
    git config --global url."https://$(cat /run/secrets/REPO_TOKEN)@github.com/".insteadOf "git@github.com:" && \
    git config --global url."https://$(cat /run/secrets/REPO_TOKEN)@github.com".insteadOf "https://github.com" && \
    python3 -m venv /venv && \
    /venv/bin/pip install --upgrade pip && \
    /venv/bin/pip install ${WHEELS_DIR}/edps*.whl && \
    rm -f /root/.gitconfig

# Second Stage: Production environment
FROM python:3.12-slim AS runner

ARG VERSION

LABEL ai.beebucket.maintainer="beebucket GmbH <hello@beebucket.ai>"
LABEL ai.beebucket.version=${VERSION}

# Install necessary packages for Open-CV
RUN apt-get update && \
    apt-get install -y python3-opencv

# Copy virtual environment from build stage
COPY --from=build /venv /venv

# Set the default Python environment
ENV PATH="/venv/bin:$PATH"

# Create a non-root user for security
RUN useradd -m appuser

# EDP internal working directory
VOLUME ["/work"]
RUN mkdir -p /work && chown -R appuser:appuser /work
ENV WORKING_DIR=/work

# Pre-download the EasyOCR model files
RUN mkdir -p /work/models/ocr
ENV EASYOCR_MODULE_PATH=/work/models/ocr
RUN python -c "import easyocr; easyocr.Reader(['en', 'de'], gpu=False, download_enabled=True)"

# Pre-download the FFmpeg binaries
RUN python -c "import static_ffmpeg; static_ffmpeg.add_paths()"

# Pre-download spaCy models
RUN python -m spacy download en_core_web_lg
RUN python -m spacy download de_core_news_lg
RUN python -m spacy download fr_core_news_lg

# Set user
USER appuser
