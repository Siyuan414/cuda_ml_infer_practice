#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# GPTQModel env setup for WSL2 + RTX 50-series (Blackwell, sm_120)
#
# What this does:
#   1. Sanity-checks WSL + nvidia-smi (driver must be >= 570.x for CUDA 12.8)
#   2. Creates a fresh Python venv at ./gptq_env
#   3. Installs PyTorch stable cu128 (Blackwell-supported)
#   4. Installs gptqmodel + the HF stack (transformers/accelerate/datasets)
#   5. Runs a quick CUDA + sm_120 verification
#
# Usage:
#   chmod +x setup_gptq_env.sh
#   ./setup_gptq_env.sh           # uses python3.11 by default
#   PY=python3.12 ./setup_gptq_env.sh
# -----------------------------------------------------------------------------
set -euo pipefail

PY="${PY:-python3.11}"
VENV_DIR="${VENV_DIR:-gptq_env}"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

log()  { printf "\033[1;34m[setup]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[warn]\033[0m  %s\n" "$*"; }
die()  { printf "\033[1;31m[err]\033[0m   %s\n" "$*" >&2; exit 1; }

# ---------- 1. host / driver checks ------------------------------------------
log "Checking WSL + NVIDIA driver..."
if ! grep -qiE "microsoft|wsl" /proc/version; then
    warn "This doesn't look like WSL. Continuing anyway."
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    die "nvidia-smi not found in WSL. Install/upgrade the NVIDIA Windows driver \
(>= 570.x for CUDA 12.8) and make sure WSL2 is up to date (\`wsl --update\` in PowerShell)."
fi

DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
log "GPU:    $GPU_NAME"
log "Driver: $DRIVER_VER"

DRIVER_MAJOR=${DRIVER_VER%%.*}
if [ "${DRIVER_MAJOR:-0}" -lt 570 ]; then
    warn "Driver $DRIVER_VER is below 570.x. Blackwell + CUDA 12.8 wheels may fail."
    warn "Update the NVIDIA driver on Windows (not in WSL)."
fi

# ---------- 2. python check --------------------------------------------------
if ! command -v "$PY" >/dev/null 2>&1; then
    die "$PY not found. Install with: sudo apt install $PY $PY-venv $PY-dev"
fi
PYV=$("$PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])')
log "Using $PY ($PYV)"

case "$PYV" in
    3.10|3.11|3.12) ;;
    *) warn "Python $PYV is outside the well-tested range (3.10-3.12) for gptqmodel." ;;
esac

# ---------- 3. system build deps (only needed if a wheel falls back to sdist) -
log "Ensuring build tooling is present (build-essential, git, ninja)..."
if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update -qq || warn "apt update failed; continuing"
    sudo apt-get install -y --no-install-recommends \
        build-essential git ninja-build "$PY-venv" "$PY-dev" \
        || warn "apt install partially failed; continuing"
fi

# ---------- 4. venv ----------------------------------------------------------
if [ -d "$VENV_DIR" ]; then
    warn "$VENV_DIR already exists. Remove it manually if you want a clean rebuild:"
    warn "    rm -rf $VENV_DIR"
else
    log "Creating venv at ./$VENV_DIR"
    "$PY" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install -U pip setuptools wheel

# ---------- 5. PyTorch (cu128, stable) ---------------------------------------
log "Installing PyTorch stable (cu128) from $TORCH_INDEX ..."
pip install --index-url "$TORCH_INDEX" torch torchvision torchaudio

# ---------- 6. HF stack + gptqmodel ------------------------------------------
log "Installing transformers / accelerate / datasets / tokenizers ..."
pip install -U \
    "transformers>=4.45" \
    "accelerate>=0.34" \
    "datasets>=3.0" \
    sentencepiece \
    protobuf \
    safetensors \
    huggingface_hub

log "Installing gptqmodel ..."
# --no-build-isolation lets gptqmodel see the torch we just installed if it
# falls back to building from source (only happens on unusual platforms).
pip install -U gptqmodel --no-build-isolation || pip install -U gptqmodel

# ---------- 7. verify --------------------------------------------------------
log "Verifying CUDA + Blackwell visibility..."
python - <<'PY'
import torch, platform
print("python   :", platform.python_version())
print("torch    :", torch.__version__)
print("cuda built against:", torch.version.cuda)
print("cuda available    :", torch.cuda.is_available())
if torch.cuda.is_available():
    i = 0
    name = torch.cuda.get_device_name(i)
    cap = torch.cuda.get_device_capability(i)
    print(f"device[{i}]: {name}  sm_{cap[0]}{cap[1]}")
    arch_list = torch.cuda.get_arch_list()
    print("torch arch list   :", arch_list)
    sm = f"sm_{cap[0]}{cap[1]}"
    if sm not in arch_list:
        print(f"WARNING: {sm} is not in torch's compiled arch list. "
              f"You may be on a torch build that predates Blackwell support.")
try:
    import gptqmodel
    print("gptqmodel:", gptqmodel.__version__)
except Exception as e:
    print("gptqmodel import FAILED:", e)
PY

log "Done. Activate with:  source $VENV_DIR/bin/activate"
