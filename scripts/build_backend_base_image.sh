#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND_DIR="${REPO_ROOT}/backend"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found" >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "docker compose plugin not found" >&2
  exit 1
fi

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  read -rsp "Enter GitHub Fine-grained PAT (read access to AutoWeaver): " GITHUB_TOKEN
  echo
fi

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "GITHUB_TOKEN is empty." >&2
  exit 1
fi

export DOCKER_BUILDKIT=1
export GITHUB_TOKEN

cd "${BACKEND_DIR}"

echo "[base-build] Building backend base image..."
docker compose -f docker-compose.base.yml build backend-base

echo "[base-build] Built image: ${BACKEND_BASE_IMAGE:-pluck/backend-base:cu128}"
