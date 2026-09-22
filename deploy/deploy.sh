#!/usr/bin/env bash
# Quick deployment and testing script for CountLoop
set -e

echo "=== Deploying CountLoop ==="

if command -v docker &> /dev/null; then
    echo "Building Docker container with GPU support..."
    docker compose -f deploy/docker-compose.yml build
    echo "Starting CountLoop service..."
    docker compose -f deploy/docker-compose.yml up -d
    echo "CountLoop is live on http://localhost:8000"
else
    echo "Docker not detected. Launching local FastAPI server..."
    pip install -e .
    python -m countloop.cli serve --host 0.0.0.0 --port 8000
fi
