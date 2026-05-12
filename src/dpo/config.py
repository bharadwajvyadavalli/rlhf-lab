"""
DPO Training Configuration
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DPOConfig:
    """Configuration for Direct Preference Optimization training."""

    # Model - start from SFT checkpoint
    model_name: str = "Qwen/Qwen2.5-1.5B"
    sft_adapter_path: str = "./outputs/sft-qwen2.5-1.5b-lora"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"

    # DPO specific
    beta: float = 0.1  # KL penalty coefficient
    loss_type: str = "sigmoid"  # sigmoid, hinge, ipo, kto_pair
    label_smoothing: float = 0.0

    # Reference model
    ref_model: Optional[str] = None  # If None, uses copy of policy model

    # LoRA
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Training
    output_dir: str = "./outputs/dpo-qwen2.5-1.5b-lora"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    max_length: int = 1024
    max_prompt_length: int = 512

    # Optimization
    bf16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"
    max_grad_norm: float = 1.0

    # Logging & Saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100
    save_total_limit: int = 3
    evaluation_strategy: str = "steps"
    load_best_model_at_end: bool = True

    # W&B
    report_to: str = "wandb"
    run_name: Optional[str] = None
    wandb_project: str = "rlhf-lab"

    # Hub
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_strategy: str = "checkpoint"

    # Data
    max_samples: Optional[int] = None
    seed: int = 42
    dataloader_num_workers: int = 4

    def __post_init__(self):
        if self.run_name is None:
            self.run_name = f"dpo-{self.model_name.split('/')[-1]}-beta{self.beta}"
