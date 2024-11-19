# First Stage: Build environment
FROM python:3.12-slim AS build
    
# Set up a virtual environment and install dependencies from pyproject.toml
ARG WHEELS_DIR="/wheels"
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip \
    --mount=type=bind,src=dist,target=${WHEELS_DIR},readonly \
    python3 -m venv /venv && \
    /venv/bin/pip install --upgrade pip && \
    /venv/bin/pip install ${WHEELS_DIR}/edp*.whl

# Second Stage: Production environment
FROM python:3.12-slim AS runner

ARG VERSION

LABEL ai.beebucket.maintainer="beebucket GmbH <hello@beebucket.ai>"
LABEL ai.beebucket.version=${VERSION}

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

# Set user
USER appuser

# Start the application
CMD ["edps"]
