#!/bin/bash
# Run Medical Summarization Fine-tuning with Config
# Usage: ./run_finetune.sh [--quick-test]

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate all

# Parse arguments
QUICK_TEST=""
if [ "$1" == "--quick-test" ]; then
    QUICK_TEST="--quick_test"
    echo "🚀 Running in quick test mode..."
fi

# Run training
python summarization/fine_tune_llama.py \
    --config config.yml \
    $QUICK_TEST

echo "✅ Training completed!"
