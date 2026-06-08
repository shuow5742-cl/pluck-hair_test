#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND_DIR="${REPO_ROOT}/backend"

BASE_COMPOSE="docker-compose.yml"
RUNTIME_COMPOSE="docker-compose.runtime.yml"
BASE_BUILD_COMPOSE="docker-compose.base.yml"

STACK_ARGS=(-f "${BASE_COMPOSE}" -f "${RUNTIME_COMPOSE}")

usage() {
  cat <<'EOF'
Usage:
  bash scripts/backend_dev.sh <command> [args]

Commands:
  check                    Verify host prerequisites (docker, gpu, camera driver)
  build-base               Build backend base image (dependencies only)
  up                       Start full stack (postgres/minio/redis/backend/backend-api)
  down                     Stop full stack
  restart                  Restart backend containers (run + api)
  logs                     Tail backend logs (run + api)
  status                   Show service status and health
  list-cameras             List connected Daheng cameras (SN, model, interface)
  setup-camera             Auto-detect camera SN, write to config, restart backend
  deploy <branch>          Full deploy: fetch branch + build-base (if needed) + start stack

Examples:
  bash scripts/backend_dev.sh check
  bash scripts/backend_dev.sh deploy test
  bash scripts/backend_dev.sh up
  bash scripts/backend_dev.sh down
  bash scripts/backend_dev.sh logs
EOF
}

# ── Prerequisites ──────────────────────────────────────────

need_tools() {
  local ok=1
  if ! command -v git >/dev/null 2>&1; then
    echo "[check] FAIL: git not found" >&2
    ok=0
  fi
  if ! command -v docker >/dev/null 2>&1; then
    echo "[check] FAIL: docker not found" >&2
    ok=0
  fi
  if ! docker compose version >/dev/null 2>&1; then
    echo "[check] FAIL: docker compose plugin not found" >&2
    ok=0
  fi
  if [[ "${ok}" -eq 0 ]]; then
    exit 1
  fi
}

check_host() {
  echo "[check] Verifying host prerequisites..."
  local ok=1

  # Docker
  if command -v docker >/dev/null 2>&1; then
    echo "[check] OK: docker $(docker --version | awk '{print $3}' | tr -d ',')"
  else
    echo "[check] FAIL: docker not found"
    ok=0
  fi

  # Docker Compose
  if docker compose version >/dev/null 2>&1; then
    echo "[check] OK: $(docker compose version)"
  else
    echo "[check] FAIL: docker compose plugin not found"
    ok=0
  fi

  # NVIDIA driver
  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu_name
    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
    echo "[check] OK: GPU ${gpu_name}"
  else
    echo "[check] FAIL: nvidia-smi not found (run bootstrap phase1)"
    ok=0
  fi

  # NVIDIA container runtime
  if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q 'nvidia'; then
    echo "[check] OK: nvidia docker runtime"
  else
    echo "[check] FAIL: nvidia docker runtime not found (run bootstrap phase2)"
    ok=0
  fi

  # Daheng camera udev rules
  if [[ -f /etc/udev/rules.d/99-daheng-camera.rules ]]; then
    echo "[check] OK: daheng udev rules"
  else
    echo "[check] WARN: daheng udev rules not found (camera may not be accessible)"
  fi

  # Daheng SDK libraries
  for lib in libgxiapi.so liblog4cplus_gx.so GxU3VTL.cti GxGVTL.cti; do
    if [[ -f "/usr/lib/${lib}" ]]; then
      echo "[check] OK: /usr/lib/${lib}"
    else
      echo "[check] WARN: /usr/lib/${lib} not found (run bootstrap phase2)"
    fi
  done

  # Galaxy config
  if [[ -d /etc/Galaxy ]]; then
    echo "[check] OK: /etc/Galaxy config"
  else
    echo "[check] WARN: /etc/Galaxy not found (run bootstrap phase2)"
  fi

  if [[ "${ok}" -eq 1 ]]; then
    echo "[check] All prerequisites met."
  else
    echo "[check] Some checks failed. Fix issues before proceeding." >&2
    exit 1
  fi
}

# ── Build ──────────────────────────────────────────────────

ensure_github_token() {
  if [[ -z "${GITHUB_TOKEN:-}" ]]; then
    read -rsp "Enter GitHub Fine-grained PAT (read access to AutoWeaver): " GITHUB_TOKEN
    echo
  fi

  if [[ -z "${GITHUB_TOKEN:-}" ]]; then
    echo "GITHUB_TOKEN is empty." >&2
    exit 1
  fi

  export GITHUB_TOKEN
}

build_base() {
  ensure_github_token
  echo "[dev] Building base image..."
  docker compose -f "${BASE_BUILD_COMPOSE}" build backend-base
  echo "[dev] Base image built: ${BACKEND_BASE_IMAGE:-pluck/backend-base:cu128}"
}

ensure_base_image() {
  local image_name="${BACKEND_BASE_IMAGE:-pluck/backend-base:cu128}"
  if ! docker image inspect "${image_name}" >/dev/null 2>&1; then
    echo "[dev] Base image not found: ${image_name}"
    build_base
  fi
}

ensure_base_image_for_deploy() {
  local image_name="${BACKEND_BASE_IMAGE:-pluck/backend-base:cu128}"
  local image_created
  local image_ts
  local deps_ts

  deps_ts="$(git -C "${REPO_ROOT}" log -1 --format=%ct -- backend/pyproject.toml backend/uv.lock 2>/dev/null || echo 0)"
  if [[ -z "${deps_ts}" ]]; then
    deps_ts=0
  fi

  if ! docker image inspect "${image_name}" >/dev/null 2>&1; then
    echo "[deploy] Base image not found: ${image_name}"
    build_base
    return
  fi

  image_created="$(docker image inspect "${image_name}" --format '{{.Created}}' 2>/dev/null || true)"
  image_ts="$(date -d "${image_created}" +%s 2>/dev/null || echo 0)"

  if (( deps_ts > image_ts )); then
    echo "[deploy] Dependency lock/config changed after base image build; rebuilding base image..."
    build_base
  else
    echo "[deploy] Base image is up to date for dependency files."
  fi
}

# ── Stack ──────────────────────────────────────────────────

ensure_x11_access() {
  if ! command -v xhost >/dev/null 2>&1; then
    echo "[dev] WARN: xhost not found, skip X11 authorization setup."
    return
  fi

  export DISPLAY="${DISPLAY:-:0}"

  if [[ -z "${XAUTHORITY:-}" ]]; then
    if [[ -f "/run/user/$(id -u)/gdm/Xauthority" ]]; then
      export XAUTHORITY="/run/user/$(id -u)/gdm/Xauthority"
    elif [[ -f "${HOME}/.Xauthority" ]]; then
      export XAUTHORITY="${HOME}/.Xauthority"
    fi
  fi

  if xhost +SI:localuser:root >/dev/null 2>&1; then
    echo "[dev] X11 access ready via SI:localuser:root (DISPLAY=${DISPLAY})."
    return
  fi

  if xhost +local: >/dev/null 2>&1; then
    echo "[dev] X11 access ready via local fallback (DISPLAY=${DISPLAY})."
    return
  fi

  echo "[dev] WARN: failed to authorize X11 (DISPLAY=${DISPLAY}, XAUTHORITY=${XAUTHORITY:-unset})."
}

stack_up() {
  ensure_base_image
  ensure_x11_access
  echo "[dev] Starting stack..."
  docker compose "${STACK_ARGS[@]}" up -d
  wait_healthy
  docker compose "${STACK_ARGS[@]}" ps
}

stack_down() {
  echo "[dev] Stopping stack..."
  docker compose "${STACK_ARGS[@]}" down
}

backend_restart() {
  ensure_base_image
  ensure_x11_access
  echo "[dev] Restarting backend containers..."
  docker compose "${STACK_ARGS[@]}" up -d backend backend-api
  docker compose "${STACK_ARGS[@]}" ps backend backend-api
}

backend_logs() {
  docker compose "${STACK_ARGS[@]}" logs -f --tail=200 backend backend-api
}

stack_status() {
  docker compose "${STACK_ARGS[@]}" ps
}

# ── Health ─────────────────────────────────────────────────

wait_healthy() {
  local services=("pluck-postgres" "pluck-minio" "pluck-redis")
  local timeout=60
  local elapsed=0

  echo "[dev] Waiting for services to be healthy..."
  for svc in "${services[@]}"; do
    while true; do
      local health
      health="$(docker inspect --format='{{.State.Health.Status}}' "${svc}" 2>/dev/null || echo "missing")"
      if [[ "${health}" == "healthy" ]]; then
        echo "[dev] OK: ${svc} is healthy"
        break
      fi
      if [[ "${elapsed}" -ge "${timeout}" ]]; then
        echo "[dev] WARN: ${svc} not healthy after ${timeout}s (status: ${health})" >&2
        break
      fi
      sleep 2
      elapsed=$((elapsed + 2))
    done
  done
}

# ── Deploy ─────────────────────────────────────────────────

deploy_branch() {
  local branch="${1:-}"
  if [[ -z "${branch}" ]]; then
    echo "branch is required: bash scripts/backend_dev.sh deploy <branch>" >&2
    exit 1
  fi

  check_host

  cd "${REPO_ROOT}"

  echo "[deploy] Fetching origin..."
  git fetch --prune origin

  if ! git show-ref --verify --quiet "refs/remotes/origin/${branch}"; then
    echo "[deploy] Remote branch not found: origin/${branch}" >&2
    exit 1
  fi

  # Stash local changes if any
  if [[ -n "$(git status --porcelain)" ]]; then
    local stash_msg="deploy-auto-stash-$(date +%Y%m%d-%H%M%S)"
    echo "[deploy] Stashing local changes: ${stash_msg}"
    git stash push -u -m "${stash_msg}" >/dev/null
  fi

  echo "[deploy] Switching to ${branch}..."
  git checkout "${branch}" 2>/dev/null || git checkout -b "${branch}" "origin/${branch}"
  git reset --hard "origin/${branch}"

  cd "${BACKEND_DIR}"

  # Build base image if missing or older than dependency file changes.
  ensure_base_image_for_deploy

  # Start full stack
  stack_up

  echo "[deploy] Done: branch ${branch} deployed and running."
}

# ── Main ───────────────────────────────────────────────────

main() {
  local cmd="${1:-}"
  if [[ -z "${cmd}" ]]; then
    usage
    exit 1
  fi

  need_tools

  cd "${REPO_ROOT}"
  if [[ ! -d .git ]]; then
    echo "Not a git repository: ${REPO_ROOT}" >&2
    exit 1
  fi

  cd "${BACKEND_DIR}"

  case "${cmd}" in
    check)
      check_host
      ;;
    build-base)
      build_base
      ;;
    up)
      stack_up
      ;;
    down)
      stack_down
      ;;
    restart)
      backend_restart
      ;;
    logs)
      backend_logs
      ;;
    status)
      stack_status
      ;;
    list-cameras)
      docker exec pluck-backend python scripts/list_cameras.py
      ;;
    setup-camera)
      echo "[dev] Detecting camera and updating config..."
      docker exec pluck-backend python scripts/setup_camera.py
      backend_restart
      ;;
    deploy)
      deploy_branch "${2:-}"
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      echo "Unknown command: ${cmd}" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
