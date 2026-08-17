#!/bin/bash
set -euo pipefail

DATASET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${DATASET_DIR}"

echo "Target directory: ${DATASET_DIR}"
echo ""

for cmd in curl tar; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "ERROR: Required command '${cmd}' is not available."
        exit 1
    fi
done

fast_download() {
    local url="$1"
    local output="$2"
    local temp="${output}.part"

    echo "Downloading: ${url}"
    curl \
        --fail \
        --location \
        --insecure \
        --continue-at - \
        --retry 10 \
        --retry-delay 3 \
        --retry-all-errors \
        --connect-timeout 20 \
        --speed-limit 1000 \
        --speed-time 30 \
        --output "${temp}" \
        "${url}"

    mv -f "${temp}" "${output}"
}

# 1. CIFAR-100 Dataset

CIFAR_DIR="${DATASET_DIR}/cifar-100-python"
CIFAR_TAR="${DATASET_DIR}/cifar-100-python.tar.gz"

if [[ -d "${CIFAR_DIR}" ]]; then
    echo "[CIFAR-100] Directory exists. Skipping."
else
    echo "[CIFAR-100] Downloading and extracting..."
    fast_download "https://data.brainchip.com/dataset-mirror/cifar100/cifar-100-python.tar.gz" "${CIFAR_TAR}"
    tar -xzf "${CIFAR_TAR}" -C "${DATASET_DIR}"
    rm -f "${CIFAR_TAR}"
    echo "[CIFAR-100] Done."
fi

echo ""

# 2. Tiny-ImageNet Archive Only

TINY_ZIP="${DATASET_DIR}/tiny-imagenet-200.zip"

if [[ -f "${TINY_ZIP}" || -d "${DATASET_DIR}/tiny-imagenet-200" ]]; then
    echo "[Tiny-ImageNet] Archive/Directory present. Skipping."
else
    echo "[Tiny-ImageNet] Downloading ZIP archive..."
    fast_download "http://cs231n.stanford.edu/tiny-imagenet-200.zip" "${TINY_ZIP}"
    echo "[Tiny-ImageNet] Download complete (Extration will be done via dataset_utils.py script)."
fi

echo ""
echo "========================================================"
echo "Download step complete! Archives ready for PyTorch."
echo "========================================================"