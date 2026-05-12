"""
UltraFeedback Dataset Preparation

Loads and processes the UltraFeedback dataset into formats suitable for:
- SFT: (prompt, chosen_response) pairs
- Reward Model / DPO: (prompt, chosen, rejected) triplets
- PPO: Prompts only for online generation
"""

from typing import Optional
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer


def load_ultrafeedback(
    tokenizer: Optional[AutoTokenizer] = None,
    split_ratios: tuple[float, float, float] = (0.9, 0.05, 0.05),
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> DatasetDict:
    """
    Load and process UltraFeedback dataset.

    Args:
        tokenizer: Optional tokenizer for applying chat template
        split_ratios: (train, val, test) split ratios
        seed: Random seed for reproducibility
        max_samples: Optional limit on total samples (for debugging)

    Returns:
        DatasetDict with train, validation, and test splits
    """
    # Load the binarized version which already has chosen/rejected
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Process into our format
    def process_example(example):
        """Convert UltraFeedback format to our standard format."""
        prompt = example["prompt"]

        # Extract chosen and rejected responses
        chosen_messages = example["chosen"]
        rejected_messages = example["rejected"]

        # Get the assistant responses (last message in each)
        chosen_response = chosen_messages[-1]["content"] if chosen_messages else ""
        rejected_response = rejected_messages[-1]["content"] if rejected_messages else ""

        # Build conversation format for tokenization
        if tokenizer is not None:
            # Format with chat template
            chosen_conv = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen_response}
            ]
            rejected_conv = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": rejected_response}
            ]

            chosen_text = tokenizer.apply_chat_template(
                chosen_conv, tokenize=False, add_generation_prompt=False
            )
            rejected_text = tokenizer.apply_chat_template(
                rejected_conv, tokenize=False, add_generation_prompt=False
            )
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            chosen_text = f"User: {prompt}\n\nAssistant: {chosen_response}"
            rejected_text = f"User: {prompt}\n\nAssistant: {rejected_response}"
            prompt_text = f"User: {prompt}\n\nAssistant:"

        return {
            "prompt": prompt,
            "prompt_formatted": prompt_text,
            "chosen": chosen_response,
            "rejected": rejected_response,
            "chosen_formatted": chosen_text,
            "rejected_formatted": rejected_text,
        }

    # Apply processing
    dataset = dataset.map(
        process_example,
        remove_columns=dataset.column_names,
        desc="Processing UltraFeedback",
        num_proc=4,
    )

    # Filter out empty responses
    dataset = dataset.filter(
        lambda x: len(x["chosen"]) > 0 and len(x["rejected"]) > 0,
        desc="Filtering empty responses",
    )

    # Create splits
    train_ratio, val_ratio, test_ratio = split_ratios
    assert abs(sum(split_ratios) - 1.0) < 1e-6, "Split ratios must sum to 1"

    # First split off test
    train_val = dataset.train_test_split(test_size=test_ratio, seed=seed)
    test_dataset = train_val["test"]

    # Then split train/val from remaining
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_val_split = train_val["train"].train_test_split(
        test_size=val_ratio_adjusted, seed=seed
    )

    return DatasetDict({
        "train": train_val_split["train"],
        "validation": train_val_split["test"],
        "test": test_dataset,
    })


def create_sft_dataset(
    dataset: DatasetDict,
    tokenizer: AutoTokenizer,
    max_length: int = 2048,
) -> DatasetDict:
    """
    Create SFT dataset from preference data.
    Uses only the chosen responses for supervised fine-tuning.

    Args:
        dataset: Processed UltraFeedback DatasetDict
        tokenizer: Tokenizer for encoding
        max_length: Maximum sequence length

    Returns:
        DatasetDict formatted for SFT training
    """
    def format_for_sft(example):
        """Format example for SFT (just the chosen response)."""
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["chosen"]}
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    sft_dataset = DatasetDict()
    for split in dataset.keys():
        sft_dataset[split] = dataset[split].map(
            format_for_sft,
            remove_columns=dataset[split].column_names,
            desc=f"Formatting {split} for SFT",
        )

    return sft_dataset


def create_preference_dataset(
    dataset: DatasetDict,
    tokenizer: AutoTokenizer,
    max_length: int = 1024,
    max_prompt_length: int = 512,
) -> DatasetDict:
    """
    Create preference dataset for DPO/Reward Model training.

    Args:
        dataset: Processed UltraFeedback DatasetDict
        tokenizer: Tokenizer for encoding
        max_length: Maximum total sequence length
        max_prompt_length: Maximum prompt length

    Returns:
        DatasetDict with prompt, chosen, rejected columns
    """
    def format_for_preference(example):
        """Format example for DPO/RM training."""
        # Format prompt with chat template
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["prompt"]}],
            tokenize=False,
            add_generation_prompt=True
        )

        return {
            "prompt": prompt,
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }

    pref_dataset = DatasetDict()
    for split in dataset.keys():
        pref_dataset[split] = dataset[split].map(
            format_for_preference,
            remove_columns=dataset[split].column_names,
            desc=f"Formatting {split} for preference learning",
        )

    return pref_dataset


def create_ppo_prompts(
    dataset: DatasetDict,
    tokenizer: AutoTokenizer,
) -> DatasetDict:
    """
    Create prompt-only dataset for PPO training.
    Responses will be generated online during training.

    Args:
        dataset: Processed UltraFeedback DatasetDict
        tokenizer: Tokenizer for encoding

    Returns:
        DatasetDict with query column (formatted prompts)
    """
    def format_for_ppo(example):
        """Format example for PPO (prompt only)."""
        query = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["prompt"]}],
            tokenize=False,
            add_generation_prompt=True
        )
        return {"query": query}

    ppo_dataset = DatasetDict()
    for split in dataset.keys():
        ppo_dataset[split] = dataset[split].map(
            format_for_ppo,
            remove_columns=dataset[split].column_names,
            desc=f"Formatting {split} for PPO",
        )

    return ppo_dataset


def get_dataset_statistics(dataset: DatasetDict) -> dict:
    """Get basic statistics about the dataset."""
    stats = {}
    for split, data in dataset.items():
        stats[split] = {
            "num_examples": len(data),
            "columns": data.column_names,
        }

        # Sample lengths if text columns exist
        if "chosen" in data.column_names:
            chosen_lengths = [len(x) for x in data["chosen"][:1000]]
            rejected_lengths = [len(x) for x in data["rejected"][:1000]]
            stats[split]["avg_chosen_length"] = sum(chosen_lengths) / len(chosen_lengths)
            stats[split]["avg_rejected_length"] = sum(rejected_lengths) / len(rejected_lengths)

    return stats


if __name__ == "__main__":
    # Quick test
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

    print("Loading UltraFeedback...")
    dataset = load_ultrafeedback(tokenizer=tokenizer, max_samples=100)

    print("\nDataset statistics:")
    stats = get_dataset_statistics(dataset)
    for split, split_stats in stats.items():
        print(f"\n{split}:")
        for key, value in split_stats.items():
            print(f"  {key}: {value}")

    print("\nSample from train split:")
    sample = dataset["train"][0]
    print(f"Prompt: {sample['prompt'][:200]}...")
    print(f"Chosen: {sample['chosen'][:200]}...")
    print(f"Rejected: {sample['rejected'][:200]}...")
