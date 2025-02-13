# First Stage: Build environment
FROM python:3.12-slim AS build

# Install git and g++ for building libsvm
RUN apt-get update && \
    apt-get install -y git build-essential

ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
    
# Set up a virtual environment and install dependencies from pyproject.toml
ARG WHEELS_DIR="/wheels"
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip \
    --mount=type=bind,src=dist,target=${WHEELS_DIR},readonly \
    python3 -m venv /venv && \
    /venv/bin/pip install --upgrade pip && \
    /venv/bin/pip install ${WHEELS_DIR}/edps*.whl

# Second Stage: Production environment
FROM python:3.12-slim AS runner

ARG VERSION

LABEL ai.beebucket.maintainer="beebucket GmbH <hello@beebucket.ai>"
LABEL ai.beebucket.version=${VERSION}

# Install necessary packages for Open-CV
RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6

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

# Set user
USER appuser
