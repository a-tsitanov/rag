#!/usr/bin/env bash
set -euo pipefail

echo "Starting Enterprise Knowledge Base..."
docker compose up -d

echo "Waiting for services to be healthy..."
docker compose wait --timeout 120 || true

echo ""
echo "Services:"
echo "  API:           http://localhost:8000"
echo "  API docs:      http://localhost:8000/docs"
echo "  Neo4j Browser: http://localhost:7474"
echo "  MinIO Console: http://localhost:9001"
echo ""
docker compose ps
