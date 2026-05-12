#!/bin/bash
# SFT Training Script
# Run on A40 48GB - ~30-60 min, ~$0.40

set -e

# Configuration
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/sft-qwen2.5-1.5b-lora}"
WANDB_PROJECT="${WANDB_PROJECT:-rlhf-lab}"

# Training hyperparameters
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-2e-4}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"

# Optional: limit samples for debugging
MAX_SAMPLES="${MAX_SAMPLES:-}"

# HuggingFace Hub (optional)
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
HUB_MODEL_ID="${HUB_MODEL_ID:-}"

echo "=============================================="
echo "SFT Training"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "Output: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE x $GRAD_ACCUM"
echo "Learning rate: $LR"
echo "LoRA: r=$LORA_R, alpha=$LORA_ALPHA"
echo "=============================================="

# Build command
CMD="python -m src.sft.train \
    --model-name $MODEL_NAME \
    --output-dir $OUTPUT_DIR \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --grad-accum $GRAD_ACCUM \
    --learning-rate $LR \
    --lora-r $LORA_R \
    --lora-alpha $LORA_ALPHA"

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
echo "SFT training complete!"
echo "Model saved to: $OUTPUT_DIR"
