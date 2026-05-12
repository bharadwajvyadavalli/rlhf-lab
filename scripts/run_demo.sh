#!/bin/bash
# Launch Gradio Demo
# Requires trained models to be available

set -e

# Configuration
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B}"
SFT_ADAPTER="${SFT_ADAPTER:-./outputs/sft-qwen2.5-1.5b-lora}"
DPO_ADAPTER="${DPO_ADAPTER:-./outputs/dpo-qwen2.5-1.5b-lora}"
PPO_ADAPTER="${PPO_ADAPTER:-./outputs/ppo-qwen2.5-1.5b-lora}"
REWARD_MODEL="${REWARD_MODEL:-./outputs/rm-qwen2.5-1.5b}"
PORT="${PORT:-7860}"
SHARE="${SHARE:-false}"

echo "=============================================="
echo "RLHF/DPO Model Comparison Demo"
echo "=============================================="
echo "Base Model: $MODEL_NAME"
echo "Port: $PORT"
echo "Share: $SHARE"
echo "=============================================="

# Check for at least one model
if [ ! -d "$SFT_ADAPTER" ] && [ ! -d "$DPO_ADAPTER" ] && [ ! -d "$PPO_ADAPTER" ]; then
    echo "WARNING: No trained models found!"
    echo "Run training first or provide model paths."
fi

# Build command
CMD="python app/gradio_demo.py \
    --base-model $MODEL_NAME \
    --sft-adapter $SFT_ADAPTER \
    --dpo-adapter $DPO_ADAPTER \
    --ppo-adapter $PPO_ADAPTER \
    --reward-model $REWARD_MODEL \
    --port $PORT"

if [ "$SHARE" = "true" ]; then
    CMD="$CMD --share"
fi

echo "Running: $CMD"
eval $CMD
