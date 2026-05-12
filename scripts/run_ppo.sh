#!/bin/bash
# PPO Training Script
# Run on A100 80GB - ~4-6 hr, ~$10.00
# This is the most resource-intensive training

set -e

# Configuration
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B}"
SFT_ADAPTER="${SFT_ADAPTER:-./outputs/sft-qwen2.5-1.5b-lora}"
REWARD_MODEL="${REWARD_MODEL:-./outputs/rm-qwen2.5-1.5b}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/ppo-qwen2.5-1.5b-lora}"
WANDB_PROJECT="${WANDB_PROJECT:-rlhf-lab}"

# PPO hyperparameters (conservative defaults)
INIT_KL_COEF="${INIT_KL_COEF:-0.2}"
TARGET_KL="${TARGET_KL:-6.0}"
LR="${LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TOTAL_EPISODES="${TOTAL_EPISODES:-10000}"

# LoRA
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"

# Optional: limit samples for debugging
MAX_SAMPLES="${MAX_SAMPLES:-}"

# HuggingFace Hub (optional)
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
HUB_MODEL_ID="${HUB_MODEL_ID:-}"

echo "=============================================="
echo "PPO Training (RLHF)"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "SFT Adapter: $SFT_ADAPTER"
echo "Reward Model: $REWARD_MODEL"
echo "Output: $OUTPUT_DIR"
echo "KL Coef: $INIT_KL_COEF (target: $TARGET_KL)"
echo "Learning rate: $LR"
echo "Batch size: $BATCH_SIZE"
echo "Total episodes: $TOTAL_EPISODES"
echo "LoRA: r=$LORA_R, alpha=$LORA_ALPHA"
echo ""
echo "WARNING: This is resource-intensive. Recommend A100 80GB."
echo "Monitor for reward hacking and KL explosion."
echo "=============================================="

# Check if reward model exists
if [ ! -d "$REWARD_MODEL" ]; then
    echo "ERROR: Reward model not found at $REWARD_MODEL"
    echo "Run reward model training first: ./scripts/run_rm.sh"
    exit 1
fi

# Build command
CMD="python -m src.ppo.train \
    --model-name $MODEL_NAME \
    --sft-adapter-path $SFT_ADAPTER \
    --reward-model-path $REWARD_MODEL \
    --output-dir $OUTPUT_DIR \
    --init-kl-coef $INIT_KL_COEF \
    --target-kl $TARGET_KL \
    --learning-rate $LR \
    --batch-size $BATCH_SIZE \
    --total-episodes $TOTAL_EPISODES \
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
echo "PPO training complete!"
echo "Model saved to: $OUTPUT_DIR"
