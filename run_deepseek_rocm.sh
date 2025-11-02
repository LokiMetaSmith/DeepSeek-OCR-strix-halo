#!/bin/bash

# ==============================================================================
# DeepSeek-OCR on ROCm - All-in-One Runner
# ==============================================================================
#
# This script handles the complete process of setting up and running the
# DeepSeek-OCR model on a ROCm-enabled system. It is designed to be
# idempotent, meaning it can be run multiple times safely.
#
#
# INSTRUCTIONS:
# 1. Activate your Python virtual environment.
#    source .venv/bin/activate
# 2. Run this script.
#    chmod +x run_deepseek_rocm.sh
#    ./run_deepseek_rocm.sh
#
# ==============================================================================

set -e

echo "### DeepSeek-OCR on ROCm - All-in-One Runner ###"
echo "Timestamp: $(date)"
echo ""

# --- Section 1: Environment Check ---
echo "--- 1. Environment Check ---"
if [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: No virtual environment is activated."
    echo "Please activate your virtual environment before running this script."
    echo "Example: source .venv/bin/activate"
    exit 1
fi
echo "Virtual environment detected: $VIRTUAL_ENV"
echo ""
echo "----------------------------------------"
echo ""

# --- Section 2: Project Repositories ---
echo "--- 2. Project Repositories ---"
if [ ! -d "DeepSeek-OCR" ]; then
    echo "Cloning DeepSeek-OCR source code..."
    git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
else
    echo "DeepSeek-OCR directory already exists."
fi
echo ""

if [ ! -d "DeepSeek-OCR-model" ]; then
    echo "Cloning DeepSeek-OCR model repository..."
    git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR DeepSeek-OCR-model
else
    echo "DeepSeek-OCR-model directory already exists."
fi
echo ""

echo "Downloading large model files (this may take a while)..."
cd DeepSeek-OCR-model
git lfs pull
cd ..
echo ""
echo "----------------------------------------"
echo ""

# --- Section 3: Dependencies ---
echo "--- 3. Dependencies ---"
echo "Installing/verifying Python dependencies..."
pip install -r DeepSeek-OCR/requirements.txt
echo ""
echo "----------------------------------------"
echo ""

# --- Section 4: Run OCR ---
echo "--- 4. Run OCR ---"
echo "Downloading test image..."
wget https://static.simonwillison.net/static/2025/ft.jpeg -O test_image.jpeg
echo ""

echo "Running OCR..."
python3 run_ocr_amd.py test_image.jpeg
echo ""
echo "----------------------------------------"
echo ""

echo "### Script Complete ###"
