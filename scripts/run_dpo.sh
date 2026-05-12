#!/bin/bash
# DPO Training Script
# Run on A40 - ~2-4 hr, ~$2.00

set -e

# Configuration
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B}"
SFT_ADAPTER="${SFT_ADAPTER:-./outputs/sft-qwen2.5-1.5b-lora}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/dpo-qwen2.5-1.5b-lora}"
WANDB_PROJECT="${WANDB_PROJECT:-rlhf-lab}"

# DPO hyperparameters
BETA="${BETA:-0.1}"
LOSS_TYPE="${LOSS_TYPE:-sigmoid}"

# Training hyperparameters
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-5e-5}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"

# Optional: limit samples for debugging
MAX_SAMPLES="${MAX_SAMPLES:-}"

# HuggingFace Hub (optional)
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
HUB_MODEL_ID="${HUB_MODEL_ID:-}"

echo "=============================================="
echo "DPO Training"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "SFT Adapter: $SFT_ADAPTER"
echo "Output: $OUTPUT_DIR"
echo "Beta: $BETA"
echo "Loss type: $LOSS_TYPE"
echo "Batch size: $BATCH_SIZE x $GRAD_ACCUM"
echo "Learning rate: $LR"
echo "LoRA: r=$LORA_R, alpha=$LORA_ALPHA"
echo "=============================================="

# Build command
CMD="python -m src.dpo.train \
    --model-name $MODEL_NAME \
    --sft-adapter-path $SFT_ADAPTER \
    --output-dir $OUTPUT_DIR \
    --beta $BETA \
    --loss-type $LOSS_TYPE \
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
echo "DPO training complete!"
echo "Model saved to: $OUTPUT_DIR"
