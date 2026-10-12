#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-shot setup for MSNet / MAROB-2500.
#
# The code builds on the BasicSR framework as shipped with NAFNet. This script fetches a
# NAFNet tree, installs the Python deps, and wires the paper-specific files (architecture,
# blur/augmentation dataset, region-metric model wrapper) into that tree so BasicSR auto-
# discovers them. After it finishes, `python smoke_test.py` verifies everything imports and
# runs -- no dataset required.
#
# Usage:
#   bash setup.sh                     # clones NAFNet into ./third_party/NAFNet
#   NAFNET_ROOT=/path/to/NAFNet bash setup.sh   # reuse an existing NAFNet checkout
#   SKIP_PIP=1 bash setup.sh          # only wire the files, don't touch pip
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"
REPO="$(pwd)"

NAFNET_URL="https://github.com/megvii-research/NAFNet.git"
NAFNET_ROOT="${NAFNET_ROOT:-$REPO/third_party/NAFNet}"

echo "== 1/3  Python dependencies =="
if [ "${SKIP_PIP:-0}" != "1" ]; then
    echo "   PyTorch (CUDA 12.1 wheels) + requirements.txt"
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
else
    echo "   SKIP_PIP=1 -> skipping pip install"
fi

echo "== 2/3  BasicSR / NAFNet framework =="
if [ ! -d "$NAFNET_ROOT/basicsr" ]; then
    echo "   cloning NAFNet -> $NAFNET_ROOT"
    git clone --depth 1 "$NAFNET_URL" "$NAFNET_ROOT"
else
    echo "   reusing existing NAFNet tree at $NAFNET_ROOT"
fi

echo "== 3/3  Wiring paper-specific files into the BasicSR tree =="
install -v -m 0644 code/archs/NAFNet_arch.py "$NAFNET_ROOT/basicsr/models/archs/NAFNet_arch.py"
install -v -m 0644 code/archs/msnet_arch.py  "$NAFNET_ROOT/basicsr/models/archs/msnet_arch.py"
# the dataset and BOTH of its helper modules go side-by-side in basicsr/data/ so the bare
# `from maritime_kernel import ...` / `from ship_mask_blur_pipeline import ...` resolve.
install -v -m 0644 code/data/maritime_dataset.py    "$NAFNET_ROOT/basicsr/data/maritime_dataset.py"
install -v -m 0644 code/data/maritime_kernel.py     "$NAFNET_ROOT/basicsr/data/maritime_kernel.py"
install -v -m 0644 code/ship_mask_blur_pipeline.py  "$NAFNET_ROOT/basicsr/data/ship_mask_blur_pipeline.py"

cat <<EOF

Done. NAFNet tree: $NAFNET_ROOT

Verify (no data needed):
    NAFNET_ROOT="$NAFNET_ROOT" python smoke_test.py

Train / evaluate (needs the dataset; see README "Data availability"):
    cd "$NAFNET_ROOT"
    python basicsr/train.py -opt "$REPO/configs/R26-MSNet-FIXED-hardevalv3-200k.yml"
    python "$REPO/scripts/eval_testsplit.py"

To synthesize blur on your own imagery, drop a PSF bank where maritime_kernel.py expects it
(or export MAROB_KERNEL_ROOT=/path/to/your/bank). See README "Data availability".
EOF
