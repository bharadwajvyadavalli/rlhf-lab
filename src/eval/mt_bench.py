"""
MT-Bench Evaluation

Multi-turn evaluation using MT-Bench (80 questions, 8 categories).
Uses GPT-4 as judge for scoring (1-10 scale).
"""

import json
import os
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm


# MT-Bench categories
MT_BENCH_CATEGORIES = [
    "writing",
    "roleplay",
    "extraction",
    "reasoning",
    "math",
    "coding",
    "knowledge",
    "common-sense",
]

# Sample MT-Bench questions (subset for demo - full set requires FastChat)
MT_BENCH_SAMPLE = [
    {
        "question_id": 1,
        "category": "writing",
        "turns": [
            "Write a short poem about a cat.",
            "Now rewrite the poem to be about a dog instead.",
        ],
    },
    {
        "question_id": 2,
        "category": "reasoning",
        "turns": [
            "What is the sum of the first 100 positive integers?",
            "Now explain Gauss's method for solving this problem.",
        ],
    },
    {
        "question_id": 3,
        "category": "coding",
        "turns": [
            "Write a Python function to check if a number is prime.",
            "Now optimize the function to be more efficient.",
        ],
    },
    {
        "question_id": 4,
        "category": "math",
        "turns": [
            "Solve the equation: 2x + 5 = 13",
            "Now solve: 3x^2 - 12x + 9 = 0",
        ],
    },
    {
        "question_id": 5,
        "category": "knowledge",
        "turns": [
            "What causes the seasons on Earth?",
            "How would seasons differ if Earth's axial tilt were 0 degrees?",
        ],
    },
]


def load_model_for_eval(
    model_name: str,
    adapter_path: Optional[str] = None,
    torch_dtype: str = "bfloat16",
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
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    """Generate a response given conversation history."""

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=generation_config)

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    return response.strip()


def evaluate_with_gpt4(
    question: dict,
    model_answers: list[str],
    api_key: Optional[str] = None,
) -> dict:
    """
    Evaluate model answers using GPT-4 as judge.

    Returns scores for each turn (1-10 scale).
    """
    try:
        from openai import OpenAI

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return {"error": "OPENAI_API_KEY not set"}

        client = OpenAI(api_key=api_key)

        scores = []
        for i, (turn, answer) in enumerate(zip(question["turns"], model_answers)):
            prompt = f"""Please rate the following response on a scale of 1-10.

Question: {turn}

Response: {answer}

Consider:
- Accuracy and correctness
- Helpfulness and relevance
- Clarity and coherence
- Depth of explanation

Provide your rating as a single number from 1-10, followed by a brief explanation.
Format: SCORE: [number]
EXPLANATION: [your explanation]"""

            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
            )

            result = response.choices[0].message.content

            # Parse score
            try:
                score_line = [l for l in result.split("\n") if "SCORE:" in l][0]
                score = int(score_line.split("SCORE:")[-1].strip().split()[0])
                scores.append(min(max(score, 1), 10))  # Clamp to 1-10
            except (IndexError, ValueError):
                scores.append(5)  # Default score if parsing fails

        return {
            "question_id": question["question_id"],
            "category": question["category"],
            "scores": scores,
            "avg_score": sum(scores) / len(scores),
        }

    except ImportError:
        return {"error": "openai package not installed"}
    except Exception as e:
        return {"error": str(e)}


def run_mt_bench(
    model_name: str,
    adapter_path: Optional[str] = None,
    output_dir: str = "./outputs/eval",
    model_label: str = "model",
    use_full_bench: bool = False,
    use_gpt4_judge: bool = True,
) -> dict:
    """
    Run MT-Bench evaluation.

    Args:
        model_name: Base model name
        adapter_path: Optional path to LoRA adapter
        output_dir: Output directory for results
        model_label: Label for this model
        use_full_bench: Use full MT-Bench (requires FastChat)
        use_gpt4_judge: Use GPT-4 for judging

    Returns:
        Dictionary with evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("MT-Bench Evaluation")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Adapter: {adapter_path}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model_for_eval(model_name, adapter_path)

    # Get questions
    questions = MT_BENCH_SAMPLE
    if use_full_bench:
        try:
            # Try to load full MT-Bench from FastChat
            from fastchat.llm_judge.common import load_questions

            questions = load_questions("mt_bench")
            print(f"Loaded {len(questions)} questions from MT-Bench")
        except ImportError:
            print("FastChat not installed, using sample questions")

    print(f"\nEvaluating on {len(questions)} questions")

    # Generate responses
    results = []
    for question in tqdm(questions, desc="Evaluating"):
        messages = []
        answers = []

        for turn in question["turns"]:
            messages.append({"role": "user", "content": turn})
            response = generate_response(model, tokenizer, messages)
            answers.append(response)
            messages.append({"role": "assistant", "content": response})

        result = {
            "question_id": question["question_id"],
            "category": question["category"],
            "turns": question["turns"],
            "answers": answers,
        }

        # Get GPT-4 judgment
        if use_gpt4_judge:
            judgment = evaluate_with_gpt4(question, answers)
            result.update(judgment)

        results.append(result)

    # Save results
    output_file = os.path.join(output_dir, f"{model_label}_mt_bench_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Aggregate scores
    if use_gpt4_judge:
        scores_by_category = {}
        all_scores = []

        for r in results:
            if "avg_score" in r:
                cat = r["category"]
                if cat not in scores_by_category:
                    scores_by_category[cat] = []
                scores_by_category[cat].append(r["avg_score"])
                all_scores.append(r["avg_score"])

        summary = {
            "overall_score": sum(all_scores) / len(all_scores) if all_scores else 0,
            "scores_by_category": {
                cat: sum(scores) / len(scores)
                for cat, scores in scores_by_category.items()
            },
            "num_questions": len(results),
        }

        print("\n" + "=" * 40)
        print("MT-Bench Results")
        print("=" * 40)
        print(f"Overall Score: {summary['overall_score']:.2f}/10")
        print("\nBy Category:")
        for cat, score in summary["scores_by_category"].items():
            print(f"  {cat}: {score:.2f}")

        return summary

    return {"num_questions": len(results), "output_file": output_file}


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
        use_gpt4: bool = typer.Option(True, help="Use GPT-4 as judge"),
    ):
        """Run MT-Bench evaluation."""
        run_mt_bench(
            model_name=model_name,
            adapter_path=adapter_path,
            output_dir=output_dir,
            model_label=model_label,
            use_gpt4_judge=use_gpt4,
        )

    app()


if __name__ == "__main__":
    main()
