#!/bin/bash
# Reward Model Training Script
# Run on A40 - ~1-2 hr, ~$1.00

set -e

# Configuration
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B}"
SFT_ADAPTER="${SFT_ADAPTER:-./outputs/sft-qwen2.5-1.5b-lora}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/rm-qwen2.5-1.5b}"
WANDB_PROJECT="${WANDB_PROJECT:-rlhf-lab}"

# Training hyperparameters
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-5}"

# Optional: limit samples for debugging
MAX_SAMPLES="${MAX_SAMPLES:-}"

# HuggingFace Hub (optional)
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
HUB_MODEL_ID="${HUB_MODEL_ID:-}"

echo "=============================================="
echo "Reward Model Training"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "SFT Adapter: $SFT_ADAPTER"
echo "Output: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE x $GRAD_ACCUM"
echo "Learning rate: $LR"
echo "=============================================="

# Build command
CMD="python -m src.rm.train \
    --model-name $MODEL_NAME \
    --output-dir $OUTPUT_DIR \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --grad-accum $GRAD_ACCUM \
    --learning-rate $LR"

if [ -d "$SFT_ADAPTER" ]; then
    CMD="$CMD --sft-adapter-path $SFT_ADAPTER"
fi

if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max-samples $MAX_SAMPLES"
fi

if [ "$PUSH_TO_HUB" = "true" ] && [ -n "$HUB_MODEL_ID" ]; then
    CMD="$CMD --push-to-hub --hub-model-id $HUB_MODEL_ID"
fi

# Run training
echo "Running: $CMD"
eval $CMD

echo ""
echo "Reward model training complete!"
echo "Model saved to: $OUTPUT_DIR"
