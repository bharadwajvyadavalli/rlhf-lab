"""
SFT Training Configuration
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SFTConfig:
    """Configuration for Supervised Fine-Tuning."""

    # Model
    model_name: str = "Qwen/Qwen2.5-1.5B"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"

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
    output_dir: str = "./outputs/sft-qwen2.5-1.5b-lora"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 2048

    # Optimization
    bf16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"
    max_grad_norm: float = 1.0

    # Logging & Saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
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
    dataset_name: str = "openbmb/UltraFeedback"
    max_samples: Optional[int] = None
    seed: int = 42
    dataloader_num_workers: int = 4

    # Packing (optional - can improve efficiency)
    packing: bool = False

    def __post_init__(self):
        if self.run_name is None:
            self.run_name = f"sft-{self.model_name.split('/')[-1]}"
