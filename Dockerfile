# Stage 1: Builder stage
FROM python:3.11-slim AS builder

# Install build dependencies and cleanup in one step
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only requirements to cache dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.11-slim

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install only runtime dependencies and cleanup in one step
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application code, excluding files specified in .dockerignore
COPY . .

# Create necessary directories
RUN mkdir -p data/{raw,processed,temp,models}

# Set the default command
CMD ["python", "main.py"]