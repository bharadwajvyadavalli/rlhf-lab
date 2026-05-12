"""
AlpacaEval 2.0 Evaluation

Runs AlpacaEval 2.0 benchmark with length-controlled win rate.
Requires OPENAI_API_KEY environment variable for GPT-4 judge.
"""

import json
import os
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm


def load_model_for_eval(
    model_name: str,
    adapter_path: Optional[str] = None,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
):
    """Load model with optional adapter for evaluation."""

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def generate_responses(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    batch_size: int = 1,
) -> list[str]:
    """Generate responses for a list of prompts."""

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    responses = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i : i + batch_size]

        # Format prompts with chat template
        formatted = []
        for prompt in batch_prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )

        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
            )

        # Decode only the new tokens
        for j, output in enumerate(outputs):
            input_len = inputs.input_ids[j].shape[0]
            response = tokenizer.decode(
                output[input_len:], skip_special_tokens=True
            )
            responses.append(response.strip())

    return responses


def run_alpaca_eval(
    model_name: str,
    adapter_path: Optional[str] = None,
    output_dir: str = "./outputs/eval",
    model_label: str = "model",
    max_samples: Optional[int] = None,
    use_alpaca_eval_cli: bool = True,
) -> dict:
    """
    Run AlpacaEval 2.0 evaluation.

    Args:
        model_name: Base model name
        adapter_path: Optional path to LoRA adapter
        output_dir: Output directory for results
        model_label: Label for this model in results
        max_samples: Optional limit on number of samples
        use_alpaca_eval_cli: Whether to use alpaca_eval CLI (requires installation)

    Returns:
        Dictionary with evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("AlpacaEval 2.0 Evaluation")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Adapter: {adapter_path}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model_for_eval(model_name, adapter_path)

    # Load AlpacaEval dataset
    print("\nLoading AlpacaEval dataset...")
    eval_dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")

    if max_samples:
        eval_dataset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))

    prompts = eval_dataset["instruction"]
    print(f"Evaluating on {len(prompts)} prompts")

    # Generate responses
    print("\nGenerating responses...")
    responses = generate_responses(model, tokenizer, prompts)

    # Prepare outputs in AlpacaEval format
    outputs = []
    for i, (instruction, response) in enumerate(zip(prompts, responses)):
        outputs.append({
            "instruction": instruction,
            "output": response,
            "generator": model_label,
            "dataset": "alpaca_eval",
        })

    # Save outputs
    output_file = os.path.join(output_dir, f"{model_label}_alpaca_eval_outputs.json")
    with open(output_file, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"\nOutputs saved to {output_file}")

    results = {"num_samples": len(prompts), "output_file": output_file}

    # Run AlpacaEval CLI if available
    if use_alpaca_eval_cli:
        try:
            import subprocess

            print("\nRunning AlpacaEval evaluation (requires OPENAI_API_KEY)...")
            eval_output_dir = os.path.join(output_dir, f"{model_label}_alpaca_eval_results")

            cmd = [
                "alpaca_eval",
                "--model_outputs", output_file,
                "--output_path", eval_output_dir,
                "--annotators_config", "alpaca_eval_gpt4_turbo_fn",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Load results
                results_file = os.path.join(eval_output_dir, "leaderboard.json")
                if os.path.exists(results_file):
                    with open(results_file) as f:
                        leaderboard = json.load(f)
                    results["leaderboard"] = leaderboard
                    print(f"\nResults: {leaderboard}")
            else:
                print(f"AlpacaEval CLI error: {result.stderr}")
                results["cli_error"] = result.stderr

        except FileNotFoundError:
            print("alpaca_eval CLI not found. Install with: pip install alpaca-eval")
            results["cli_error"] = "alpaca_eval not installed"
        except Exception as e:
            print(f"Error running alpaca_eval: {e}")
            results["cli_error"] = str(e)

    return results


def main():
    """CLI entry point."""
    import typer

    app = typer.Typer()

    @app.command()
    def evaluate(
        model_name: str = typer.Option("Qwen/Qwen2.5-1.5B", help="Base model name"),
        adapter_path: Optional[str] = typer.Option(None, help="Path to adapter"),
        output_dir: str = typer.Option("./outputs/eval", help="Output directory"),
        model_label: str = typer.Option("model", help="Model label"),
        max_samples: Optional[int] = typer.Option(None, help="Max samples"),
    ):
        """Run AlpacaEval evaluation."""
        run_alpaca_eval(
            model_name=model_name,
            adapter_path=adapter_path,
            output_dir=output_dir,
            model_label=model_label,
            max_samples=max_samples,
        )

    app()


if __name__ == "__main__":
    main()
