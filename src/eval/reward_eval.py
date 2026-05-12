"""
Reward Model Evaluation

Evaluates reward model accuracy on held-out preference data.
Reports train/test accuracy gap to detect overfitting.
"""

import os
from typing import Optional

import torch
from datasets import DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import numpy as np


def evaluate_reward_model(
    reward_model_path: str,
    test_dataset: Optional[DatasetDict] = None,
    base_model_name: str = "Qwen/Qwen2.5-1.5B",
    max_samples: Optional[int] = None,
    batch_size: int = 8,
) -> dict:
    """
    Evaluate reward model on preference pairs.

    Args:
        reward_model_path: Path to trained reward model
        test_dataset: Optional test dataset. If None, loads UltraFeedback test split.
        base_model_name: Base model name for tokenizer
        max_samples: Optional limit on samples
        batch_size: Evaluation batch size

    Returns:
        Dictionary with evaluation metrics
    """
    print("=" * 60)
    print("Reward Model Evaluation")
    print("=" * 60)
    print(f"Model: {reward_model_path}")
    print("=" * 60)

    # Load reward model
    print("\nLoading reward model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        reward_model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load test data if not provided
    if test_dataset is None:
        from ..data.prepare_ultrafeedback import load_ultrafeedback

        print("\nLoading UltraFeedback test split...")
        full_dataset = load_ultrafeedback(tokenizer=tokenizer, max_samples=max_samples)
        test_dataset = full_dataset["test"]

    if max_samples:
        test_dataset = test_dataset.select(range(min(max_samples, len(test_dataset))))

    print(f"Evaluating on {len(test_dataset)} samples")

    # Evaluate
    correct = 0
    total = 0
    chosen_rewards = []
    rejected_rewards = []
    margins = []

    device = next(model.parameters()).device

    for i in tqdm(range(0, len(test_dataset), batch_size), desc="Evaluating"):
        batch = test_dataset[i : i + batch_size]

        # Format chosen and rejected
        chosen_texts = []
        rejected_texts = []

        for j in range(len(batch["prompt"])):
            prompt = batch["prompt"][j]
            chosen = batch["chosen"][j]
            rejected = batch["rejected"][j]

            # Format with chat template
            chosen_msg = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen},
            ]
            rejected_msg = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": rejected},
            ]

            chosen_texts.append(
                tokenizer.apply_chat_template(chosen_msg, tokenize=False)
            )
            rejected_texts.append(
                tokenizer.apply_chat_template(rejected_msg, tokenize=False)
            )

        # Tokenize
        chosen_inputs = tokenizer(
            chosen_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        rejected_inputs = tokenizer(
            rejected_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        # Get rewards
        with torch.no_grad():
            chosen_outputs = model(**chosen_inputs)
            rejected_outputs = model(**rejected_inputs)

            chosen_reward = chosen_outputs.logits.squeeze(-1).cpu().numpy()
            rejected_reward = rejected_outputs.logits.squeeze(-1).cpu().numpy()

        # Track metrics
        for c, r in zip(chosen_reward, rejected_reward):
            chosen_rewards.append(float(c))
            rejected_rewards.append(float(r))
            margins.append(float(c - r))

            if c > r:
                correct += 1
            total += 1

    # Compute metrics
    accuracy = correct / total
    chosen_rewards = np.array(chosen_rewards)
    rejected_rewards = np.array(rejected_rewards)
    margins = np.array(margins)

    results = {
        "accuracy": accuracy,
        "num_samples": total,
        "chosen_reward_mean": float(chosen_rewards.mean()),
        "chosen_reward_std": float(chosen_rewards.std()),
        "rejected_reward_mean": float(rejected_rewards.mean()),
        "rejected_reward_std": float(rejected_rewards.std()),
        "margin_mean": float(margins.mean()),
        "margin_std": float(margins.std()),
        "margin_positive_rate": float((margins > 0).mean()),
    }

    print("\n" + "=" * 40)
    print("Evaluation Results")
    print("=" * 40)
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Chosen Reward: {results['chosen_reward_mean']:.3f} +/- {results['chosen_reward_std']:.3f}")
    print(f"Rejected Reward: {results['rejected_reward_mean']:.3f} +/- {results['rejected_reward_std']:.3f}")
    print(f"Margin: {results['margin_mean']:.3f} +/- {results['margin_std']:.3f}")

    return results


def compare_train_test_accuracy(
    reward_model_path: str,
    base_model_name: str = "Qwen/Qwen2.5-1.5B",
    max_samples_per_split: int = 1000,
) -> dict:
    """
    Compare reward model accuracy on train vs test split.
    Large gap indicates overfitting.
    """
    from ..data.prepare_ultrafeedback import load_ultrafeedback

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_ultrafeedback(tokenizer=tokenizer)

    # Evaluate on both splits
    train_results = evaluate_reward_model(
        reward_model_path,
        test_dataset=dataset["train"].select(range(min(max_samples_per_split, len(dataset["train"])))),
        base_model_name=base_model_name,
    )

    test_results = evaluate_reward_model(
        reward_model_path,
        test_dataset=dataset["test"].select(range(min(max_samples_per_split, len(dataset["test"])))),
        base_model_name=base_model_name,
    )

    gap = train_results["accuracy"] - test_results["accuracy"]

    print("\n" + "=" * 40)
    print("Train/Test Comparison")
    print("=" * 40)
    print(f"Train Accuracy: {train_results['accuracy']:.2%}")
    print(f"Test Accuracy: {test_results['accuracy']:.2%}")
    print(f"Gap: {gap:.2%}")

    if gap > 0.15:
        print("\nWARNING: Large train/test gap suggests overfitting!")
    elif gap > 0.10:
        print("\nNOTE: Moderate train/test gap, monitor for overfitting.")

    return {
        "train_accuracy": train_results["accuracy"],
        "test_accuracy": test_results["accuracy"],
        "gap": gap,
        "train_results": train_results,
        "test_results": test_results,
    }


def main():
    """CLI entry point."""
    import typer

    app = typer.Typer()

    @app.command()
    def evaluate(
        reward_model_path: str = typer.Option(
            "./outputs/rm-qwen2.5-1.5b", help="Path to reward model"
        ),
        base_model_name: str = typer.Option(
            "Qwen/Qwen2.5-1.5B", help="Base model for tokenizer"
        ),
        max_samples: Optional[int] = typer.Option(None, help="Max samples"),
        compare_splits: bool = typer.Option(
            False, help="Compare train/test accuracy"
        ),
    ):
        """Evaluate reward model."""
        if compare_splits:
            compare_train_test_accuracy(
                reward_model_path,
                base_model_name=base_model_name,
                max_samples_per_split=max_samples or 1000,
            )
        else:
            evaluate_reward_model(
                reward_model_path,
                base_model_name=base_model_name,
                max_samples=max_samples,
            )

    app()


if __name__ == "__main__":
    main()
