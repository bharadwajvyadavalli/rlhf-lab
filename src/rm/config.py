"""
Reward Model Training Configuration
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RMConfig:
    """Configuration for Reward Model Training."""

    # Model - start from SFT checkpoint
    model_name: str = "Qwen/Qwen2.5-1.5B"
    sft_adapter_path: Optional[str] = "./outputs/sft-qwen2.5-1.5b-lora"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"

    # Training
    output_dir: str = "./outputs/rm-qwen2.5-1.5b"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_length: int = 2048

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
    metric_for_best_model: str = "accuracy"
    greater_is_better: bool = True

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
            self.run_name = f"rm-{self.model_name.split('/')[-1]}"
