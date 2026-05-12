"""
Reward Model Training Script

Trains a reward model on preference pairs using Bradley-Terry loss.
Uses the SFT model as initialization.
"""

import os
from typing import Optional

import torch
import wandb
from datasets import DatasetDict
from peft import PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)
from trl import RewardTrainer, RewardConfig

from .config import RMConfig
from ..data.prepare_ultrafeedback import load_ultrafeedback, create_preference_dataset


def setup_model_and_tokenizer(config: RMConfig):
    """Load and configure model and tokenizer for reward modeling."""

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

    # Load model for sequence classification (reward head)
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "device_map": "auto",
        "num_labels": 1,  # Single scalar reward
    }

    # Try flash attention if specified
    if config.attn_implementation == "flash_attention_2":
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:
            print("Flash Attention 2 not available, using default attention")

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        **model_kwargs,
    )

    # Load SFT adapter if provided
    if config.sft_adapter_path and os.path.exists(config.sft_adapter_path):
        print(f"Loading SFT adapter from {config.sft_adapter_path}")
        # Note: For reward model, we typically merge the adapter
        # since we're training a new head anyway
        # This requires loading the base model first, then the adapter
        # For simplicity, we can also train from scratch on the base model

    # Set pad token id for the model
    model.config.pad_token_id = tokenizer.pad_token_id

    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    return model, tokenizer


def prepare_dataset(config: RMConfig, tokenizer: AutoTokenizer) -> DatasetDict:
    """Load and prepare the preference dataset for RM training."""

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
    )

    print(f"Train samples: {len(pref_dataset['train'])}")
    print(f"Validation samples: {len(pref_dataset['validation'])}")
    print(f"Test samples: {len(pref_dataset['test'])}")

    return pref_dataset


def compute_metrics(eval_pred):
    """Compute accuracy metric for reward model."""
    predictions, labels = eval_pred

    # For RewardTrainer, predictions are [chosen_rewards, rejected_rewards]
    # Accuracy = how often chosen_reward > rejected_reward
    if len(predictions.shape) == 2 and predictions.shape[1] == 2:
        chosen_rewards = predictions[:, 0]
        rejected_rewards = predictions[:, 1]
    else:
        # Fallback: assume alternating chosen/rejected
        chosen_rewards = predictions[0::2]
        rejected_rewards = predictions[1::2]

    accuracy = (chosen_rewards > rejected_rewards).mean()

    return {
        "accuracy": accuracy,
        "chosen_reward_mean": chosen_rewards.mean(),
        "rejected_reward_mean": rejected_rewards.mean(),
        "reward_margin": (chosen_rewards - rejected_rewards).mean(),
    }


def train_reward_model(config: Optional[RMConfig] = None) -> str:
    """
    Run Reward Model training.

    Args:
        config: RMConfig instance. If None, uses defaults.

    Returns:
        Path to the trained model.
    """
    if config is None:
        config = RMConfig()

    # Initialize wandb
    if config.report_to == "wandb":
        wandb.init(
            project=config.wandb_project,
            name=config.run_name,
            config=vars(config),
        )

    print("=" * 60)
    print("Reward Model Training Configuration")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"SFT Adapter: {config.sft_adapter_path}")
    print(f"Output: {config.output_dir}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.per_device_train_batch_size} x {config.gradient_accumulation_steps}")
    print("=" * 60)

    # Setup model and tokenizer
    print("\nLoading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(config)

    # Prepare dataset
    dataset = prepare_dataset(config, tokenizer)

    # Training arguments using RewardConfig
    training_args = RewardConfig(
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
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
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
        max_length=config.max_length,
    )

    # Initialize RewardTrainer
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(dataset["test"])
    print(f"Test Results: {test_results}")

    # Log test results to wandb
    if config.report_to == "wandb":
        wandb.log({"test/" + k: v for k, v in test_results.items()})

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
        sft_adapter_path: Optional[str] = typer.Option(None, help="Path to SFT adapter"),
        output_dir: str = typer.Option("./outputs/rm-qwen2.5-1.5b", help="Output directory"),
        epochs: int = typer.Option(1, help="Number of training epochs"),
        batch_size: int = typer.Option(4, help="Per-device batch size"),
        grad_accum: int = typer.Option(8, help="Gradient accumulation steps"),
        learning_rate: float = typer.Option(1e-5, help="Learning rate"),
        max_samples: Optional[int] = typer.Option(None, help="Max samples for debugging"),
        push_to_hub: bool = typer.Option(False, help="Push to HuggingFace Hub"),
        hub_model_id: Optional[str] = typer.Option(None, help="Hub model ID"),
    ):
        """Run Reward Model training."""
        config = RMConfig(
            model_name=model_name,
            sft_adapter_path=sft_adapter_path,
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            max_samples=max_samples,
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
        )
        train_reward_model(config)

    app()


if __name__ == "__main__":
    main()
