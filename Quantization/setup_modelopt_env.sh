#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Set up a separate venv for NVIDIA ModelOpt at ./modelopt_env
#
# We use a separate venv (not gptq_env) because nvidia-modelopt pins
# transformers<5 while gptqmodel pins transformers>=5.4 — incompatible.
# Two venvs avoids the constant fight over transformers version.
#
# Usage:
#   chmod +x setup_modelopt_env.sh
#   ./setup_modelopt_env.sh
#
# After it finishes, activate with:
#   source modelopt_env/bin/activate
#   python tinyllama_modelopt_awq.py
# -----------------------------------------------------------------------------
set -euo pipefail

PY="${PY:-python3.11}"
VENV_DIR="${VENV_DIR:-modelopt_env}"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

log()  { printf "\033[1;34m[setup]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[warn]\033[0m  %s\n" "$*"; }
die()  { printf "\033[1;31m[err]\033[0m   %s\n" "$*" >&2; exit 1; }

# ---------- python check -----------------------------------------------------
if ! command -v "$PY" >/dev/null 2>&1; then
    die "$PY not found. Install with: sudo apt install $PY $PY-venv $PY-dev"
fi
log "Using $PY ($("$PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])'))"

# ---------- create venv -------------------------------------------------------
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

# ---------- pytorch (cu128) ---------------------------------------------------
log "Installing PyTorch stable (cu128)..."
pip install --index-url "$TORCH_INDEX" torch torchvision torchaudio

# ---------- modelopt + minimum deps the script uses ---------------------------
# nvidia-modelopt[hf,torch] pulls transformers, accelerate, peft, deepspeed,
# diffusers, etc. — the full stack ModelOpt needs for HF model handling.
log "Installing nvidia-modelopt[hf,torch]..."
pip install -U "nvidia-modelopt[hf,torch]"

log "Installing extra deps used by tinyllama_modelopt_awq.py..."
pip install -U datasets sentencepiece protobuf

# ---------- verify ------------------------------------------------------------
log "Verifying..."
python - <<'PY'
import torch, platform
print("python   :", platform.python_version())
print("torch    :", torch.__version__)
print("cuda     :", torch.version.cuda, "available:", torch.cuda.is_available())
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f"device   : {torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}")

import modelopt
import modelopt.torch.quantization as mtq
print("modelopt :", modelopt.__version__)
print("INT4_AWQ_CFG keys:", list(mtq.INT4_AWQ_CFG.keys())[:6], "...")

import transformers
print("transformers:", transformers.__version__)
PY

log "Done. Activate and run the modelopt task with:"
log "    source $VENV_DIR/bin/activate"
log "    python tinyllama_modelopt_awq.py"
