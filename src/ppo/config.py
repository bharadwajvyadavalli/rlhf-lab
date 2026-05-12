"""
PPO Training Configuration
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PPOConfig:
    """Configuration for Proximal Policy Optimization training."""

    # Policy Model - start from SFT checkpoint
    model_name: str = "Qwen/Qwen2.5-1.5B"
    sft_adapter_path: str = "./outputs/sft-qwen2.5-1.5b-lora"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"

    # Reward Model
    reward_model_path: str = "./outputs/rm-qwen2.5-1.5b"

    # LoRA for policy
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # PPO Hyperparameters (Conservative)
    learning_rate: float = 1e-5
    batch_size: int = 4  # Total batch size
    mini_batch_size: int = 1  # For PPO updates
    ppo_epochs: int = 4  # Updates per batch
    gamma: float = 1.0  # Discount factor
    lam: float = 0.95  # GAE lambda
    cliprange: float = 0.2  # Policy clip range
    cliprange_value: float = 0.2  # Value clip range
    vf_coef: float = 0.1  # Value function coefficient

    # KL Control (Critical for stability)
    kl_penalty: str = "kl"  # "kl", "abs", "mse", "full"
    init_kl_coef: float = 0.2  # Initial KL coefficient (conservative)
    target_kl: float = 6.0  # Target KL divergence
    adap_kl_ctrl: bool = True  # Adaptive KL coefficient
    horizon: int = 10000  # For adaptive KL

    # Generation
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True

    # Reward Processing
    reward_clip: float = 10.0  # Clip rewards to [-clip, clip]
    use_score_scaling: bool = True
    use_score_norm: bool = True
    score_clip: float = 10.0

    # Training
    output_dir: str = "./outputs/ppo-qwen2.5-1.5b-lora"
    total_episodes: int = 10000
    save_steps: int = 100
    log_steps: int = 10
    eval_steps: int = 100

    # Optimization
    bf16: bool = True
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0

    # Early stopping
    kl_threshold: float = 15.0  # Stop if KL exceeds this
    reward_baseline: Optional[float] = None  # For detecting reward hacking

    # W&B
    report_to: str = "wandb"
    run_name: Optional[str] = None
    wandb_project: str = "rlhf-lab"

    # Hub
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None

    # Data
    max_samples: Optional[int] = None
    seed: int = 42

    def __post_init__(self):
        if self.run_name is None:
            self.run_name = f"ppo-{self.model_name.split('/')[-1]}-kl{self.init_kl_coef}"
