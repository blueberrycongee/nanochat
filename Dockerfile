# Nanochat Dockerfile
# Multi-stage build for efficient deployment

# Stage 1: Build stage with Rust for rustbpe
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install uv for fast Python package management
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY rustbpe/ ./rustbpe/

# Install Python dependencies
RUN uv venv .venv && \
    . .venv/bin/activate && \
    uv sync --extra cpu

# Build rustbpe
RUN . .venv/bin/activate && \
    pip install maturin && \
    maturin develop --release

# Stage 2: Runtime stage
FROM python:3.11-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY nanochat/ ./nanochat/
COPY scripts/ ./scripts/
COPY tasks/ ./tasks/

# Copy rustbpe built library
COPY --from=builder /app/.venv/lib/python3.11/site-packages/rustbpe* /app/.venv/lib/python3.11/site-packages/

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - runs the web server
# Note: Mount your model checkpoint at /app/checkpoints
CMD ["python", "-m", "scripts.chat_web", "--host", "0.0.0.0", "--port", "8000"]
