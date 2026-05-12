#!/bin/bash
# Evaluation Script
# Run on A10G - ~1-2 hr, ~$0.80 + GPT-4 API costs

set -e

# Configuration
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B}"
SFT_ADAPTER="${SFT_ADAPTER:-./outputs/sft-qwen2.5-1.5b-lora}"
DPO_ADAPTER="${DPO_ADAPTER:-./outputs/dpo-qwen2.5-1.5b-lora}"
PPO_ADAPTER="${PPO_ADAPTER:-./outputs/ppo-qwen2.5-1.5b-lora}"
REWARD_MODEL="${REWARD_MODEL:-./outputs/rm-qwen2.5-1.5b}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/eval}"

# Evaluation options
RUN_ALPACA="${RUN_ALPACA:-true}"
RUN_MTBENCH="${RUN_MTBENCH:-true}"
RUN_RM_EVAL="${RUN_RM_EVAL:-true}"

# Optional: limit samples for debugging
MAX_SAMPLES="${MAX_SAMPLES:-}"

echo "=============================================="
echo "Model Evaluation Suite"
echo "=============================================="
echo "Base Model: $MODEL_NAME"
echo "Output: $OUTPUT_DIR"
echo ""
echo "Evaluations:"
echo "  AlpacaEval: $RUN_ALPACA"
echo "  MT-Bench: $RUN_MTBENCH"
echo "  RM Eval: $RUN_RM_EVAL"
echo ""
echo "NOTE: AlpacaEval and MT-Bench require OPENAI_API_KEY"
echo "=============================================="

mkdir -p "$OUTPUT_DIR"

# Reward Model Evaluation
if [ "$RUN_RM_EVAL" = "true" ] && [ -d "$REWARD_MODEL" ]; then
    echo ""
    echo "=== Evaluating Reward Model ==="
    python -m src.eval.reward_eval \
        --reward-model-path "$REWARD_MODEL" \
        --base-model-name "$MODEL_NAME" \
        --compare-splits \
        ${MAX_SAMPLES:+--max-samples $MAX_SAMPLES}
fi

# AlpacaEval for each model
if [ "$RUN_ALPACA" = "true" ]; then
    echo ""
    echo "=== AlpacaEval Evaluation ==="

    # SFT baseline
    if [ -d "$SFT_ADAPTER" ]; then
        echo "Evaluating SFT model..."
        python -m src.eval.alpaca_eval \
            --model-name "$MODEL_NAME" \
            --adapter-path "$SFT_ADAPTER" \
            --output-dir "$OUTPUT_DIR" \
            --model-label "sft" \
            ${MAX_SAMPLES:+--max-samples $MAX_SAMPLES}
    fi

    # DPO
    if [ -d "$DPO_ADAPTER" ]; then
        echo "Evaluating DPO model..."
        python -m src.eval.alpaca_eval \
            --model-name "$MODEL_NAME" \
            --adapter-path "$DPO_ADAPTER" \
            --output-dir "$OUTPUT_DIR" \
            --model-label "dpo" \
            ${MAX_SAMPLES:+--max-samples $MAX_SAMPLES}
    fi

    # PPO
    if [ -d "$PPO_ADAPTER" ]; then
        echo "Evaluating PPO model..."
        python -m src.eval.alpaca_eval \
            --model-name "$MODEL_NAME" \
            --adapter-path "$PPO_ADAPTER" \
            --output-dir "$OUTPUT_DIR" \
            --model-label "ppo" \
            ${MAX_SAMPLES:+--max-samples $MAX_SAMPLES}
    fi
fi

# MT-Bench for each model
if [ "$RUN_MTBENCH" = "true" ]; then
    echo ""
    echo "=== MT-Bench Evaluation ==="

    # SFT baseline
    if [ -d "$SFT_ADAPTER" ]; then
        echo "Evaluating SFT model..."
        python -m src.eval.mt_bench \
            --model-name "$MODEL_NAME" \
            --adapter-path "$SFT_ADAPTER" \
            --output-dir "$OUTPUT_DIR" \
            --model-label "sft"
    fi

    # DPO
    if [ -d "$DPO_ADAPTER" ]; then
        echo "Evaluating DPO model..."
        python -m src.eval.mt_bench \
            --model-name "$MODEL_NAME" \
            --adapter-path "$DPO_ADAPTER" \
            --output-dir "$OUTPUT_DIR" \
            --model-label "dpo"
    fi

    # PPO
    if [ -d "$PPO_ADAPTER" ]; then
        echo "Evaluating PPO model..."
        python -m src.eval.mt_bench \
            --model-name "$MODEL_NAME" \
            --adapter-path "$PPO_ADAPTER" \
            --output-dir "$OUTPUT_DIR" \
            --model-label "ppo"
    fi
fi

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
