#!/usr/bin/env bash
#
# Set up Earth2Studio + StormScope on a RunPod pod, with everything that
# matters living on the persistent network volume at /workspace.
#
# Base image: RunPod PyTorch template (or NGC pytorch). Either works -- the
# base image's torch is never used; this script builds its own venv.
#
# Why /workspace for everything: on RunPod the container disk is wiped when
# the pod stops. Site-packages, the uv-managed Python, and the 9.4 GB model
# cache all go on the network volume so a stop/start is cheap.
#
# Run under tmux -- if natten has to build from source it is long, and a
# dropped SSH session would kill it:
#
#   tmux new -s setup
#   bash /workspace/setup_stormscope.sh
#   # ctrl-b d to detach, tmux attach -t setup to come back
#
set -euo pipefail

WS=/workspace
VENV="$WS/venv"

# ---------------------------------------------------------------------------
# VERSION PINS -- these three must stay consistent with each other.
#
# TORCH_CUDA is capped by the pod's NVIDIA driver:
#   driver r570 -> CUDA 12.8 max   |   driver r580+ -> CUDA 13 ok
# Earth2Studio >= 0.14.0 defaults to CUDA 13 wheels, which will NOT run on
# an r570 driver, hence the explicit pre-pin below.
#
# NATTEN_PIN must match TORCH_VERSION+TORCH_CUDA exactly. NATTEN only ships
# prebuilt wheels for the two most recent official torch builds, so the
# newest natten release often does NOT cover an older torch. Check
# https://natten.org/install/ for the current matrix before changing these.
#
# Getting this wrong fails SILENTLY: natten installs without libnatten,
# falls back to its FlexAttention backend, and then tries to materialize a
# ~98 GiB dense attention mask mid-rollout. The assert below catches it.
# ---------------------------------------------------------------------------
E2S_TAG="0.17.0"
TORCH_VERSION="2.11.0"
TORCH_CUDA="cu128"
NATTEN_PIN="natten==0.21.6+torch2110cu128"

echo "=============================================================="
echo " Earth2Studio ${E2S_TAG} + StormScope -- RunPod setup"
echo " torch ${TORCH_VERSION}+${TORCH_CUDA} | ${NATTEN_PIN}"
echo "=============================================================="

if [ ! -d "$WS" ]; then
    echo "ERROR: $WS does not exist. Attach a network volume with mount"
    echo "       path /workspace before running this."
    exit 1
fi

# --------------------------------------------------------------------------
# 0. sanity: GPU present, and driver new enough for the pinned CUDA
# --------------------------------------------------------------------------
echo
echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version \
    --format=csv || { echo "ERROR: no GPU visible."; exit 1; }

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "compute capability: $CC   driver: $DRV"
case "$CC" in
    8.0) echo "NOTE: A100. On NVIDIA's tested list for StormScope, but below"
         echo "      the >=8.9 Earth2Studio recommends." ;;
esac

# --------------------------------------------------------------------------
# 1. system packages (container disk -- wiped on pod stop, so this reruns
#    every session; it is fast)
# --------------------------------------------------------------------------
echo
echo "--- system packages ---"
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    git make curl cmake python3-dev tmux \
    libeccodes-tools libeccodes-dev

# NGC containers pin pip via PIP_CONSTRAINT, which breaks resolution here.
unset PIP_CONSTRAINT || true

# --------------------------------------------------------------------------
# 2. uv, with its Python and cache on the volume
# --------------------------------------------------------------------------
echo
echo "--- uv ---"
export UV_INSTALL_DIR="$WS/.uv-bin"
export UV_PYTHON_INSTALL_DIR="$WS/.uv-python"
export UV_CACHE_DIR="$WS/.uv-cache"
mkdir -p "$UV_INSTALL_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_CACHE_DIR"

if [ ! -x "$UV_INSTALL_DIR/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$UV_INSTALL_DIR:$PATH"
uv --version

# --------------------------------------------------------------------------
# 3. virtualenv on the volume
# --------------------------------------------------------------------------
echo
echo "--- virtualenv at $VENV ---"
if [ ! -d "$VENV" ]; then
    uv venv --python 3.12 "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python --version

# --------------------------------------------------------------------------
# 4. torch FIRST, pinned to the CUDA the driver supports.
#
#    Installing this before earth2studio means the resolver sees the
#    requirement already satisfied and will not pull a CUDA 13 build.
# --------------------------------------------------------------------------
echo
echo "--- torch ${TORCH_VERSION}+${TORCH_CUDA} ---"
uv pip install "torch==${TORCH_VERSION}" torchvision \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

python - <<EOF
import torch, sys
want = "${TORCH_VERSION}+${TORCH_CUDA}"
print("torch:", torch.__version__, "| cuda:", torch.version.cuda,
      "| available:", torch.cuda.is_available())
if not torch.__version__.startswith(want):
    print(f"ERROR: expected {want}, got {torch.__version__}", file=sys.stderr)
    sys.exit(1)
if not torch.cuda.is_available():
    print("ERROR: CUDA not available", file=sys.stderr); sys.exit(1)
print("device:", torch.cuda.get_device_name(0))
EOF

# --------------------------------------------------------------------------
# 5. earth2grid (needs --no-build-isolation), then Earth2Studio
# --------------------------------------------------------------------------
echo
echo "--- earth2grid ---"
uv pip install --no-build-isolation \
  "earth2grid @ git+https://github.com/NVlabs/earth2grid@11dcf1b0787a7eb6a8497a3a5a5e1fdcc31232d3"

echo
echo "--- earth2studio[data,stormscope] ${E2S_TAG} ---"
time uv pip install \
  "earth2studio[data,stormscope]@git+https://github.com/NVIDIA/earth2studio.git@${E2S_TAG}"

# --------------------------------------------------------------------------
# 6. natten WITH libnatten -- the critical step.
#
#    Earth2Studio's dependency pulls plain `natten` from PyPI, which only
#    builds the CUDA kernels if it can see a CUDA device at build time.
#    Inside uv's isolated build env it usually cannot, so it silently ships
#    without libnatten and every attention call falls back to FlexAttention.
#    Force-reinstall the prebuilt wheel matching our exact torch build.
# --------------------------------------------------------------------------
echo
echo "--- natten + libnatten (${NATTEN_PIN}) ---"
uv pip install --force-reinstall "${NATTEN_PIN}" -f https://whl.natten.org

python - <<'EOF'
import sys, natten, torch
ok = getattr(natten, "HAS_LIBNATTEN", False)
print(f"natten {natten.__version__} | HAS_LIBNATTEN: {ok}")
if not ok:
    print("", file=sys.stderr)
    print("FATAL: libnatten missing. natten will fall back to its "
          "FlexAttention backend,", file=sys.stderr)
    print("which materializes a dense attention mask and OOMs "
          "(~98 GiB) mid-rollout.", file=sys.stderr)
    print(f"Check https://natten.org/install/ for a wheel matching "
          f"torch {torch.__version__}, then update NATTEN_PIN.",
          file=sys.stderr)
    sys.exit(1)
print("libnatten OK -- fused FNA kernels available")
EOF

# --------------------------------------------------------------------------
# 7. torch must have survived the last two installs
# --------------------------------------------------------------------------
echo
echo "--- re-verify torch after all installs ---"
python - <<EOF
import torch, sys
want = "${TORCH_VERSION}+${TORCH_CUDA}"
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
if not torch.__version__.startswith(want) or not torch.cuda.is_available():
    print(f"ERROR: torch was changed to {torch.__version__}. Reinstall with:",
          file=sys.stderr)
    print(f"  uv pip install --force-reinstall torch=={want.split('+')[0]} "
          f"--index-url https://download.pytorch.org/whl/${TORCH_CUDA}",
          file=sys.stderr)
    sys.exit(1)
EOF

echo
echo "--- post-processing extras ---"
uv pip install cartopy xarray netcdf4 matplotlib

# --------------------------------------------------------------------------
# 8. environment file, sourced on every future login
# --------------------------------------------------------------------------
cat > "$WS/env.sh" <<'EOF'
# source /workspace/env.sh
export UV_INSTALL_DIR=/workspace/.uv-bin
export UV_PYTHON_INSTALL_DIR=/workspace/.uv-python
export UV_CACHE_DIR=/workspace/.uv-cache
export PATH="$UV_INSTALL_DIR:$PATH"

# Keep the 9.4 GB StormScope package and all cached GOES/MRMS/GLM granules
# on the network volume so they survive pod stop/terminate.
export EARTH2STUDIO_CACHE=/workspace/e2s-cache
export EARTH2STUDIO_PACKAGE_TIMEOUT=1800

# Cartopy coastline shapefiles, so they download once rather than per pod.
export CARTOPY_USER_BACKGROUNDS=/workspace/.cartopy
export XDG_DATA_HOME=/workspace/.local/share

source /workspace/venv/bin/activate
cd /workspace
EOF

grep -q 'workspace/env.sh' ~/.bashrc 2>/dev/null || \
    echo '[ -f /workspace/env.sh ] && source /workspace/env.sh' >> ~/.bashrc

export EARTH2STUDIO_CACHE="$WS/e2s-cache"
export EARTH2STUDIO_PACKAGE_TIMEOUT=1800
export XDG_DATA_HOME="$WS/.local/share"
mkdir -p "$EARTH2STUDIO_CACHE" "$WS/outputs" "$WS/triggers" "$WS/.local/share"

# --------------------------------------------------------------------------
# 9. verify the model API and list the real registry keys
# --------------------------------------------------------------------------
echo
echo "--- verify earth2studio ---"
python - <<'EOF'
import earth2studio
from earth2studio.models.px.stormscope import StormScopeGOES, StormScopeMRMS
print("earth2studio:", earth2studio.__version__)
print("GOES variants:", list(StormScopeGOES.list_available_models().keys()))
print("MRMS variants:", list(StormScopeMRMS.list_available_models().keys()))
EOF

# --------------------------------------------------------------------------
# 10. pre-stage the 9.4 GB model package
# --------------------------------------------------------------------------
echo
echo "--- downloading StormScope checkpoints (~9.4 GB, cached after first run) ---"
time python - <<'EOF'
from earth2studio.models.px.stormscope import StormScopeBase
pkg = StormScopeBase.load_default_package()
for f in ["registry.json", "lat.npy", "lon.npy", "topo.npy",
          "nexrad_proximity.npy", "mrms_coverage_mask.npy",
          "goes_means.npy", "goes_stds.npy",
          "mrms_means.npy", "mrms_stds.npy"]:
    pkg.resolve(f)
for kind in ["goes", "mrms"]:
    for i in range(3):
        pkg.resolve(f"checkpoints/{kind}/3km_10min/expert_{i}.mdlus")
print("package cached")
EOF
du -sh "$EARTH2STUDIO_CACHE" || true

# --------------------------------------------------------------------------
# 11. live probe of the public data sources
# --------------------------------------------------------------------------
echo
echo "--- probing GOES + MRMS + GLM on public S3 ---"
python - <<'EOF'
from datetime import datetime, timedelta, timezone
from earth2studio.data import GOES, MRMS, GOESGLMGrid

now = datetime.now(timezone.utc) - timedelta(minutes=25)
t = now.replace(minute=(now.minute // 10) * 10, second=0, microsecond=0)
print("probe time:", t)

ok = True
for name, fn in [
    ("GOES goes19/C", lambda: GOES(satellite="goes19", scan_mode="C")(
        time=t, variable=["abi13c"]).shape),
    ("MRMS refc    ", lambda: MRMS()(time=t, variable=["refc"]).shape),
    ("GLM gridded  ", lambda: GOESGLMGrid(satellite="east")(
        time=t, variable=["glm_density"]).shape),
]:
    try:
        print(f"  {name}:", tuple(fn()))
    except Exception as e:
        ok = False
        print(f"  {name}: FAILED {type(e).__name__}: {e}")

if not ok:
    print("\n  404s mean the frames are not on S3 yet -- back the init off")
    print("  further, or use an archived retrospective case.")
EOF

echo
echo "=============================================================="
echo " Done. New shells auto-source /workspace/env.sh."
echo
echo " Smoke test (1 member, 1 step -- proves the whole pipeline):"
echo "   python /workspace/stormscope_cape.py run \\"
echo "       --init 2025-07-16T19:00 --lead-minutes 10 \\"
echo "       --members 1 --num-steps 30"
echo
echo " Ensemble run once you know the per-step timing:"
echo "   python /workspace/stormscope_cape.py run \\"
echo "       --init 2025-07-16T19:00 --lead-minutes 120 \\"
echo "       --members 8 --batch-size 4"
echo
echo " Verify (add --zoom-km 60 to tighten the plots on the Cape):"
echo "   python /workspace/verify_stormscope_cape.py \\"
echo "       /workspace/outputs/stormscope_cape_*.nc --zoom-km 100"
echo "=============================================================="
