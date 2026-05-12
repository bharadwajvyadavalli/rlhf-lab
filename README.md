# RLHF-Lab: Complete RLHF/DPO Pipeline

A complete implementation of RLHF (Reinforcement Learning from Human Feedback) and DPO (Direct Preference Optimization) training pipelines for small language models.

## Overview

This project implements a full post-training pipeline:

1. **SFT (Supervised Fine-Tuning)**: Fine-tune base model on instruction data
2. **Reward Model**: Train a preference model on human feedback
3. **DPO**: Direct Preference Optimization (simpler, no reward model needed)
4. **PPO**: Proximal Policy Optimization with learned reward model
5. **Evaluation**: AlpacaEval 2.0 and MT-Bench benchmarks

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rlhf-lab.git
cd rlhf-lab

# Install dependencies
pip install -e .

# Or install from requirements
pip install -r requirements.txt

# Login to HuggingFace and Weights & Biases
huggingface-cli login
wandb login
```

### Run Full Pipeline

```bash
# Run everything sequentially
./scripts/run_all.sh

# Or run individual stages
./scripts/run_sft.sh
./scripts/run_rm.sh
./scripts/run_dpo.sh
./scripts/run_ppo.sh
./scripts/run_eval.sh
```

### Launch Demo

```bash
./scripts/run_demo.sh
# Or with sharing enabled
SHARE=true ./scripts/run_demo.sh
```

## Project Structure

```
rlhf-lab/
├── src/
│   ├── data/                    # Data preparation
│   │   └── prepare_ultrafeedback.py
│   ├── sft/                     # Supervised Fine-Tuning
│   │   ├── config.py
│   │   └── train.py
│   ├── rm/                      # Reward Model
│   │   ├── config.py
│   │   └── train.py
│   ├── dpo/                     # Direct Preference Optimization
│   │   ├── config.py
│   │   └── train.py
│   ├── ppo/                     # Proximal Policy Optimization
│   │   ├── config.py
│   │   └── train.py
│   └── eval/                    # Evaluation
│       ├── alpaca_eval.py
│       ├── mt_bench.py
│       └── reward_eval.py
├── app/
│   └── gradio_demo.py           # Side-by-side comparison demo
├── scripts/                     # Training scripts
├── configs/                     # Configuration files
├── outputs/                     # Trained models (gitignored)
└── requirements.txt
```

### Complete File Listing

**Training Modules:**
| File | Purpose |
|------|---------|
| `src/data/prepare_ultrafeedback.py` | Load & process UltraFeedback dataset |
| `src/sft/config.py` | SFT hyperparameters |
| `src/sft/train.py` | SFT training script |
| `src/rm/config.py` | Reward model hyperparameters |
| `src/rm/train.py` | Reward model training script |
| `src/dpo/config.py` | DPO hyperparameters |
| `src/dpo/train.py` | DPO training script |
| `src/ppo/config.py` | PPO hyperparameters |
| `src/ppo/train.py` | PPO training script |
| `src/eval/alpaca_eval.py` | AlpacaEval 2.0 evaluation |
| `src/eval/mt_bench.py` | MT-Bench evaluation |
| `src/eval/reward_eval.py` | Reward model accuracy evaluation |

**Infrastructure:**
| File | Purpose |
|------|---------|
| `app/gradio_demo.py` | Side-by-side model comparison demo |
| `configs/base_config.yaml` | Shared configuration |
| `requirements.txt` | Python dependencies |
| `pyproject.toml` | Project metadata & dependencies |

**Shell Scripts (for RunPod):**
| Script | Purpose |
|--------|---------|
| `scripts/run_sft.sh` | Run SFT training |
| `scripts/run_rm.sh` | Run reward model training |
| `scripts/run_dpo.sh` | Run DPO training |
| `scripts/run_ppo.sh` | Run PPO training |
| `scripts/run_eval.sh` | Run evaluation suite |
| `scripts/run_all.sh` | Run full pipeline (all stages) |
| `scripts/run_demo.sh` | Launch Gradio demo |

## Running on RunPod

### Step-by-Step Instructions

**1. Create a RunPod Instance**
- Go to [RunPod.io](https://runpod.io)
- Select GPU: **A40 48GB** for SFT/RM/DPO, **A100 80GB** for PPO
- Use PyTorch template (CUDA 12.1+)

**2. Clone and Setup**
```bash
# Clone repository
git clone https://github.com/yourusername/rlhf-lab.git
cd rlhf-lab

# Install dependencies
pip install -e .

# Login to services
huggingface-cli login
wandb login

# Set OpenAI key for evals (optional)
export OPENAI_API_KEY="your-key-here"
```

**3. Run Training (Option A: Full Pipeline)**
```bash
# Runs SFT -> RM -> DPO -> PPO -> Eval sequentially
./scripts/run_all.sh
```

**4. Run Training (Option B: Individual Stages)**
```bash
# Stage 1: SFT (A40, ~30-60 min)
./scripts/run_sft.sh

# Stage 2: Reward Model (A40, ~1-2 hr)
./scripts/run_rm.sh

# Stage 3: DPO (A40, ~2-4 hr)
./scripts/run_dpo.sh

# Stage 4: PPO (A100 recommended, ~4-6 hr)
./scripts/run_ppo.sh

# Stage 5: Evaluation
./scripts/run_eval.sh
```

**5. Quick Test Run (Debug Mode)**
```bash
# Limit samples to test the pipeline quickly
MAX_SAMPLES=100 ./scripts/run_sft.sh
MAX_SAMPLES=100 ./scripts/run_rm.sh
MAX_SAMPLES=100 ./scripts/run_dpo.sh
MAX_SAMPLES=100 TOTAL_EPISODES=100 ./scripts/run_ppo.sh
```

**6. Launch Demo (after training)**
```bash
./scripts/run_demo.sh

# Or with public URL
SHARE=true ./scripts/run_demo.sh
```

**7. Push to HuggingFace Hub**
```bash
# Push each model after training
HUB_MODEL_ID=yourusername/qwen2.5-1.5b-sft PUSH_TO_HUB=true ./scripts/run_sft.sh
HUB_MODEL_ID=yourusername/qwen2.5-1.5b-rm PUSH_TO_HUB=true ./scripts/run_rm.sh
HUB_MODEL_ID=yourusername/qwen2.5-1.5b-dpo PUSH_TO_HUB=true ./scripts/run_dpo.sh
HUB_MODEL_ID=yourusername/qwen2.5-1.5b-ppo PUSH_TO_HUB=true ./scripts/run_ppo.sh
```

### Output Artifacts

After training, you'll have:
```
outputs/
├── sft-qwen2.5-1.5b-lora/    # SFT LoRA adapter
├── rm-qwen2.5-1.5b/          # Reward model (full)
├── dpo-qwen2.5-1.5b-lora/    # DPO LoRA adapter
├── ppo-qwen2.5-1.5b-lora/    # PPO LoRA adapter
└── eval/                      # Evaluation results
```

### Monitoring Training

- **Weights & Biases**: Check wandb.ai for live metrics
- **Checkpoints**: Saved every 500 steps (configurable)
- **Logs**: Console output shows loss, accuracy, KL divergence

## Configuration

### Base Model
- **Qwen2.5-1.5B** (default): Fast iteration, lower memory requirements
- Llama-3.2-3B: Stronger base, higher compute

### Dataset
- **UltraFeedback** (~64k pairs): GPT-4 annotated, high quality
- Anthropic HH-RLHF (~160k pairs): Classical benchmark
- HelpSteer2 (~10k samples): Multi-attribute ratings

## Training Details

### SFT (Supervised Fine-Tuning)

Fine-tunes the base model on chosen responses from preference data.

```bash
# Default settings
./scripts/run_sft.sh

# Custom settings
MODEL_NAME=Qwen/Qwen2.5-1.5B \
LR=2e-4 \
BATCH_SIZE=4 \
./scripts/run_sft.sh
```

**Key Hyperparameters:**
- LoRA: r=64, alpha=128
- Learning rate: 2e-4
- Batch size: 4 x 4 gradient accumulation
- Epochs: 1

**Hardware:** A40 48GB, ~30-60 min

### Reward Model

Trains a preference model using Bradley-Terry loss.

```bash
./scripts/run_rm.sh
```

**Key Metrics:**
- Target held-out accuracy: >70%
- Monitor train/test gap for overfitting

**Hardware:** A40, ~1-2 hr

### DPO (Direct Preference Optimization)

Directly optimizes policy on preference pairs without reward model.

```bash
# Default beta=0.1
./scripts/run_dpo.sh

# Experiment with different beta
BETA=0.05 ./scripts/run_dpo.sh
```

**Key Hyperparameters:**
- Beta (KL penalty): 0.1
- Loss type: sigmoid
- Learning rate: 5e-5

**Hardware:** A40, ~2-4 hr

### PPO (Proximal Policy Optimization)

Full RLHF with learned reward model. This is the most challenging part.

```bash
./scripts/run_ppo.sh
```

**Stability Measures:**
- Conservative KL coefficient (0.2)
- Reward clipping to [-10, 10]
- Gradient clipping (max_norm=1.0)
- Frequent checkpointing (every 100 steps)
- KL divergence monitoring (stop if > 15)

**Common Failure Modes:**
1. **Reward Hacking**: Reward increases but quality degrades
2. **Mode Collapse**: Repetitive/degenerate outputs
3. **KL Explosion**: Policy diverges too far from reference

**Hardware:** A100 80GB recommended, ~4-6 hr

## Evaluation

### AlpacaEval 2.0

Length-controlled win rate vs baseline using GPT-4 as judge.

```bash
# Requires OPENAI_API_KEY
RUN_ALPACA=true ./scripts/run_eval.sh
```

### MT-Bench

Multi-turn evaluation across 8 categories (80 questions).

```bash
RUN_MTBENCH=true ./scripts/run_eval.sh
```

### Reward Model Accuracy

```bash
python -m src.eval.reward_eval --compare-splits
```

## Results

| Model | AlpacaEval LC Win% | MT-Bench | RM Accuracy |
|-------|-------------------|----------|-------------|
| SFT   | baseline          | -        | N/A         |
| DPO   | -                 | -        | N/A         |
| PPO   | -                 | -        | N/A         |

*(Fill in after running experiments)*

## Cost Estimates

Running on RunPod community cloud:

| Phase | Hardware | Time | Cost |
|-------|----------|------|------|
| SFT | A40 48GB | 30-60 min | ~$0.40 |
| RM | A40 | 1-2 hr | ~$1.00 |
| DPO | A40 | 2-4 hr | ~$2.00 |
| PPO | A100 80GB | 4-6 hr | ~$10.00 |
| Eval | A10G | 1-2 hr | ~$0.80 |

**Total:** ~$15-20 compute + ~$20-40 GPT-4 API for evals

## Gradio Demo

Compare all three models side-by-side:

```bash
python app/gradio_demo.py --share
```

Features:
- Side-by-side SFT/DPO/PPO comparison
- Reward model scoring
- Adjustable generation parameters

## Pushing to HuggingFace Hub

```bash
# Set your HF username
HUB_MODEL_ID=yourusername/qwen2.5-1.5b-sft-ultrafeedback \
PUSH_TO_HUB=true \
./scripts/run_sft.sh
```

## Known Issues & Learnings

*(Document your findings here after running experiments)*

1. **DPO Hyperparameters**: Beta=0.1 works well for UltraFeedback. Lower values (0.05) may help with harder preferences.

2. **PPO Stability**: Starting with higher KL coefficient prevents early divergence. Reduce gradually if training is stable.

3. **Reward Model Overfitting**: If train accuracy >> test accuracy, consider early stopping or regularization.

## References

- [DPO Paper](https://arxiv.org/abs/2305.18290) - Rafailov et al., 2023
- [InstructGPT Paper](https://arxiv.org/abs/2203.02155) - Ouyang et al., 2022
- [TRL Documentation](https://huggingface.co/docs/trl)
- [Zephyr Training](https://huggingface.co/blog/zephyr)

## License

MIT License - see [LICENSE](LICENSE)

---

Built as part of a bridging project for post-training engineering roles.
