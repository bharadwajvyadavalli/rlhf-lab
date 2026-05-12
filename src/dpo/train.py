"""
Direct Preference Optimization (DPO) Training Script

Trains a language model using DPO on preference pairs.
Uses the SFT model as initialization and reference.
"""

import os
from typing import Optional

import torch
import wandb
from datasets import DatasetDict
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import DPOTrainer, DPOConfig as TRLDPOConfig

from .config import DPOConfig
from ..data.prepare_ultrafeedback import load_ultrafeedback, create_preference_dataset


def setup_model_and_tokenizer(config: DPOConfig):
    """Load and configure model and tokenizer for DPO."""

    # Determine torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(config.torch_dtype, torch.bfloat16)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load base model
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "device_map": "auto",
    }

    # Try flash attention if specified
    if config.attn_implementation == "flash_attention_2":
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:
            print("Flash Attention 2 not available, using default attention")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs,
    )

    # Load SFT adapter if it exists
    if config.sft_adapter_path and os.path.exists(config.sft_adapter_path):
        print(f"Loading SFT adapter from {config.sft_adapter_path}")
        model = PeftModel.from_pretrained(
            model,
            config.sft_adapter_path,
            is_trainable=False,
        )
        # Merge adapter weights for DPO (we'll add new LoRA for DPO)
        model = model.merge_and_unload()
        print("SFT adapter merged into base model")

    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    return model, tokenizer


def setup_lora(model, config: DPOConfig):
    """Apply LoRA configuration for DPO training."""

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def prepare_dataset(config: DPOConfig, tokenizer: AutoTokenizer) -> DatasetDict:
    """Load and prepare the preference dataset for DPO."""

    # Load raw dataset
    print("Loading UltraFeedback dataset...")
    raw_dataset = load_ultrafeedback(
        tokenizer=tokenizer,
        max_samples=config.max_samples,
        seed=config.seed,
    )

    # Convert to preference format
    print("Converting to preference format...")
    pref_dataset = create_preference_dataset(
        raw_dataset,
        tokenizer=tokenizer,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
    )

    print(f"Train samples: {len(pref_dataset['train'])}")
    print(f"Validation samples: {len(pref_dataset['validation'])}")

    return pref_dataset


def train_dpo(config: Optional[DPOConfig] = None) -> str:
    """
    Run DPO training.

    Args:
        config: DPOConfig instance. If None, uses defaults.

    Returns:
        Path to the trained model/adapter.
    """
    if config is None:
        config = DPOConfig()

    # Initialize wandb
    if config.report_to == "wandb":
        wandb.init(
            project=config.wandb_project,
            name=config.run_name,
            config=vars(config),
        )

    print("=" * 60)
    print("DPO Training Configuration")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"SFT Adapter: {config.sft_adapter_path}")
    print(f"Output: {config.output_dir}")
    print(f"Beta (KL penalty): {config.beta}")
    print(f"Loss type: {config.loss_type}")
    print(f"LoRA r: {config.lora_r}, alpha: {config.lora_alpha}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.per_device_train_batch_size} x {config.gradient_accumulation_steps}")
    print("=" * 60)

    # Setup model and tokenizer
    print("\nLoading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(config)

    # Apply LoRA for DPO
    if config.use_lora:
        print("\nApplying LoRA for DPO...")
        model = setup_lora(model, config)

    # Prepare dataset
    dataset = prepare_dataset(config, tokenizer)

    # DPO training arguments
    training_args = TRLDPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy=config.evaluation_strategy,
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        report_to=config.report_to,
        run_name=config.run_name,
        optim=config.optim,
        max_grad_norm=config.max_grad_norm,
        dataloader_num_workers=config.dataloader_num_workers,
        gradient_checkpointing=config.gradient_checkpointing,
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id,
        hub_strategy=config.hub_strategy,
        seed=config.seed,
        # DPO specific
        beta=config.beta,
        loss_type=config.loss_type,
        label_smoothing=config.label_smoothing,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
    )

    # Initialize DPO Trainer
    # Note: DPOTrainer will create reference model automatically
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Will create copy of model as reference
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    # Train
    print("\nStarting DPO training...")
    trainer.train()

    # Save final model
    print("\nSaving model...")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # Push to hub if configured
    if config.push_to_hub and config.hub_model_id:
        print(f"\nPushing to Hub: {config.hub_model_id}")
        trainer.push_to_hub()

    if config.report_to == "wandb":
        wandb.finish()

    print(f"\nDPO training complete! Model saved to: {config.output_dir}")
    return config.output_dir


def main():
    """CLI entry point."""
    import typer

    app = typer.Typer()

    @app.command()
    def run(
        model_name: str = typer.Option("Qwen/Qwen2.5-1.5B", help="Base model name"),
        sft_adapter_path: str = typer.Option(
            "./outputs/sft-qwen2.5-1.5b-lora", help="Path to SFT adapter"
        ),
        output_dir: str = typer.Option("./outputs/dpo-qwen2.5-1.5b-lora", help="Output directory"),
        beta: float = typer.Option(0.1, help="DPO beta (KL penalty)"),
        loss_type: str = typer.Option("sigmoid", help="DPO loss type"),
        epochs: int = typer.Option(1, help="Number of training epochs"),
        batch_size: int = typer.Option(2, help="Per-device batch size"),
        grad_accum: int = typer.Option(8, help="Gradient accumulation steps"),
        learning_rate: float = typer.Option(5e-5, help="Learning rate"),
        lora_r: int = typer.Option(64, help="LoRA rank"),
        lora_alpha: int = typer.Option(128, help="LoRA alpha"),
        max_samples: Optional[int] = typer.Option(None, help="Max samples for debugging"),
        push_to_hub: bool = typer.Option(False, help="Push to HuggingFace Hub"),
        hub_model_id: Optional[str] = typer.Option(None, help="Hub model ID"),
    ):
        """Run DPO training."""
        config = DPOConfig(
            model_name=model_name,
            sft_adapter_path=sft_adapter_path,
            output_dir=output_dir,
            beta=beta,
            loss_type=loss_type,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            max_samples=max_samples,
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
        )
        train_dpo(config)

    app()


if __name__ == "__main__":
    main()
