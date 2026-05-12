"""
Supervised Fine-Tuning (SFT) Training Script

Fine-tunes a base model on chosen responses from UltraFeedback
using LoRA adapters.
"""

import os
from typing import Optional

import torch
import wandb
from datasets import DatasetDict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from .config import SFTConfig
from ..data.prepare_ultrafeedback import load_ultrafeedback, create_sft_dataset


def setup_model_and_tokenizer(config: SFTConfig):
    """Load and configure model and tokenizer."""

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

    # Load model
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

    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    return model, tokenizer


def setup_lora(model, config: SFTConfig):
    """Apply LoRA configuration to model."""

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


def prepare_dataset(config: SFTConfig, tokenizer: AutoTokenizer) -> DatasetDict:
    """Load and prepare the SFT dataset."""

    # Load raw dataset
    print("Loading UltraFeedback dataset...")
    raw_dataset = load_ultrafeedback(
        tokenizer=tokenizer,
        max_samples=config.max_samples,
        seed=config.seed,
    )

    # Convert to SFT format
    print("Converting to SFT format...")
    sft_dataset = create_sft_dataset(
        raw_dataset,
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
    )

    print(f"Train samples: {len(sft_dataset['train'])}")
    print(f"Validation samples: {len(sft_dataset['validation'])}")

    return sft_dataset


def train_sft(config: Optional[SFTConfig] = None) -> str:
    """
    Run SFT training.

    Args:
        config: SFTConfig instance. If None, uses defaults.

    Returns:
        Path to the trained model/adapter.
    """
    if config is None:
        config = SFTConfig()

    # Initialize wandb
    if config.report_to == "wandb":
        wandb.init(
            project=config.wandb_project,
            name=config.run_name,
            config=vars(config),
        )

    print("=" * 60)
    print("SFT Training Configuration")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Output: {config.output_dir}")
    print(f"LoRA r: {config.lora_r}, alpha: {config.lora_alpha}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.per_device_train_batch_size} x {config.gradient_accumulation_steps}")
    print("=" * 60)

    # Setup model and tokenizer
    print("\nLoading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(config)

    # Apply LoRA
    if config.use_lora:
        print("\nApplying LoRA...")
        model = setup_lora(model, config)

    # Prepare dataset
    dataset = prepare_dataset(config, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
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
    )

    # Response template for completion-only training
    # This ensures we only compute loss on assistant responses
    response_template = "<|im_start|>assistant\n"

    # Create data collator for completion-only LM
    # This masks the prompt tokens so loss is only on completions
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        max_seq_length=config.max_seq_length,
        packing=config.packing,
        dataset_text_field="text",
    )

    # Train
    print("\nStarting training...")
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

    print(f"\nTraining complete! Model saved to: {config.output_dir}")
    return config.output_dir


def main():
    """CLI entry point."""
    import typer

    app = typer.Typer()

    @app.command()
    def run(
        model_name: str = typer.Option("Qwen/Qwen2.5-1.5B", help="Base model name"),
        output_dir: str = typer.Option("./outputs/sft-qwen2.5-1.5b-lora", help="Output directory"),
        epochs: int = typer.Option(1, help="Number of training epochs"),
        batch_size: int = typer.Option(4, help="Per-device batch size"),
        grad_accum: int = typer.Option(4, help="Gradient accumulation steps"),
        learning_rate: float = typer.Option(2e-4, help="Learning rate"),
        lora_r: int = typer.Option(64, help="LoRA rank"),
        lora_alpha: int = typer.Option(128, help="LoRA alpha"),
        max_samples: Optional[int] = typer.Option(None, help="Max samples for debugging"),
        push_to_hub: bool = typer.Option(False, help="Push to HuggingFace Hub"),
        hub_model_id: Optional[str] = typer.Option(None, help="Hub model ID"),
    ):
        """Run SFT training."""
        config = SFTConfig(
            model_name=model_name,
            output_dir=output_dir,
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
        train_sft(config)

    app()


if __name__ == "__main__":
    main()
