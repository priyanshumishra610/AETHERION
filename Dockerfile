# AETHERION - Synthetic Sovereign AI System
# Multi-stage Docker build for production deployment

# Stage 1: Base image with Python and system dependencies
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV AETHERION_ENV=production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libopenjp2-7-dev \
    libtiff5-dev \
    libwebp-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libxcb1-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r aetherion && useradd -r -g aetherion aetherion

# Stage 2: Development dependencies
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    pre-commit

# Stage 3: Production build
FROM base as production

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p \
    logs \
    data \
    plugins \
    personalities \
    keeper_dashboard/frontend/build \
    && chown -R aetherion:aetherion /app

# Stage 4: Frontend build (if needed)
FROM node:18-alpine as frontend-builder

WORKDIR /app/frontend

# Copy frontend files
COPY keeper_dashboard/frontend/package*.json ./

# Install frontend dependencies
RUN npm ci --only=production

# Copy frontend source
COPY keeper_dashboard/frontend/src ./src
COPY keeper_dashboard/frontend/public ./public
COPY keeper_dashboard/frontend/tailwind.config.js .
COPY keeper_dashboard/frontend/postcss.config.js .

# Build frontend
RUN npm run build

# Stage 5: Final production image
FROM production as final

# Copy built frontend from frontend-builder
COPY --from=frontend-builder /app/frontend/build /app/keeper_dashboard/frontend/build

# Security hardening
RUN chmod -R 755 /app \
    && chmod -R 600 /app/*.json \
    && chmod -R 600 /app/*.txt \
    && chmod -R 600 /app/*.log \
    && chmod -R 600 /app/*.pem \
    && chmod -R 600 /app/*.sig

# Create volume mount points
VOLUME ["/app/logs", "/app/data", "/app/plugins", "/app/personalities"]

# Expose ports
EXPOSE 8000
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Switch to non-root user
USER aetherion

# Set entrypoint
ENTRYPOINT ["python", "scripts/start_aetherion.py"]

# Default command
CMD ["--production"] 