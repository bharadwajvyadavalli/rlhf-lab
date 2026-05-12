#!/bin/bash
# Run Full RLHF Pipeline
# This script runs all training stages sequentially

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "RLHF/DPO Full Pipeline"
echo "=============================================="
echo ""
echo "This will run:"
echo "  1. SFT Training"
echo "  2. Reward Model Training"
echo "  3. DPO Training"
echo "  4. PPO Training"
echo "  5. Evaluation"
echo ""
echo "Estimated time: 8-14 hours"
echo "Estimated cost: ~$15-20 on RunPod"
echo "=============================================="
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Stage 1: SFT
echo ""
echo ">>> Stage 1/5: SFT Training"
echo ""
bash "$SCRIPT_DIR/run_sft.sh"

# Stage 2: Reward Model
echo ""
echo ">>> Stage 2/5: Reward Model Training"
echo ""
bash "$SCRIPT_DIR/run_rm.sh"

# Stage 3: DPO
echo ""
echo ">>> Stage 3/5: DPO Training"
echo ""
bash "$SCRIPT_DIR/run_dpo.sh"

# Stage 4: PPO
echo ""
echo ">>> Stage 4/5: PPO Training"
echo ""
bash "$SCRIPT_DIR/run_ppo.sh"

# Stage 5: Evaluation
echo ""
echo ">>> Stage 5/5: Evaluation"
echo ""
bash "$SCRIPT_DIR/run_eval.sh"

echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo ""
echo "Artifacts:"
echo "  SFT:    ./outputs/sft-qwen2.5-1.5b-lora"
echo "  RM:     ./outputs/rm-qwen2.5-1.5b"
echo "  DPO:    ./outputs/dpo-qwen2.5-1.5b-lora"
echo "  PPO:    ./outputs/ppo-qwen2.5-1.5b-lora"
echo "  Eval:   ./outputs/eval"
echo ""
echo "Next steps:"
echo "  1. Review evaluation results in ./outputs/eval"
echo "  2. Launch demo: python app/gradio_demo.py"
echo "  3. Push to HuggingFace Hub (optional)"
echo "=============================================="
