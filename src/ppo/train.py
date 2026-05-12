"""
Proximal Policy Optimization (PPO) Training Script

Trains a language model using PPO with a learned reward model.
This is the most challenging part of the RLHF pipeline.

Key stability measures:
- Conservative KL coefficient
- Reward clipping
- Gradient clipping
- Frequent checkpointing
- KL divergence monitoring
"""

import os
from typing import Optional
from collections import deque

import torch
import wandb
from datasets import DatasetDict
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
)
from trl import PPOTrainer, PPOConfig as TRLPPOConfig, AutoModelForCausalLMWithValueHead

from .config import PPOConfig
from ..data.prepare_ultrafeedback import load_ultrafeedback, create_ppo_prompts


class PPOTrainingManager:
    """
    Manages PPO training with stability monitoring and diagnostics.
    """

    def __init__(self, config: PPOConfig):
        self.config = config
        self.reward_history = deque(maxlen=100)
        self.kl_history = deque(maxlen=100)
        self.step = 0
        self.best_reward = float("-inf")

    def check_stability(self, stats: dict) -> tuple[bool, str]:
        """
        Check training stability and return (is_stable, reason).
        """
        # Check KL divergence
        kl = stats.get("objective/kl", 0)
        if kl > self.config.kl_threshold:
            return False, f"KL divergence ({kl:.2f}) exceeded threshold ({self.config.kl_threshold})"

        # Check for reward hacking (reward increasing but KL exploding)
        if len(self.reward_history) >= 50 and len(self.kl_history) >= 50:
            recent_rewards = list(self.reward_history)[-20:]
            recent_kl = list(self.kl_history)[-20:]

            # Detect reward hacking pattern
            if (
                sum(recent_rewards) / len(recent_rewards) > self.best_reward + 0.5
                and sum(recent_kl) / len(recent_kl) > 5.0
            ):
                return False, "Potential reward hacking detected (reward increasing with high KL)"

        return True, "Training stable"

    def log_diagnostics(self, stats: dict, trainer: PPOTrainer):
        """Log detailed diagnostics for debugging."""
        diagnostics = {
            "step": self.step,
            "reward/mean": stats.get("ppo/returns/mean", 0),
            "reward/std": stats.get("ppo/returns/std", 0),
            "kl/mean": stats.get("objective/kl", 0),
            "kl/coef": stats.get("objective/kl_coef", 0),
            "policy/entropy": stats.get("objective/entropy", 0),
            "policy/clipfrac": stats.get("ppo/policy/clipfrac", 0),
            "value/loss": stats.get("ppo/val/error", 0),
            "lr": stats.get("ppo/lr", 0),
        }

        # Track history
        self.reward_history.append(diagnostics["reward/mean"])
        self.kl_history.append(diagnostics["kl/mean"])

        # Update best reward
        if diagnostics["reward/mean"] > self.best_reward:
            self.best_reward = diagnostics["reward/mean"]

        return diagnostics


def setup_policy_model(config: PPOConfig, tokenizer: AutoTokenizer):
    """Load and configure the policy model with value head."""

    # Determine torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(config.torch_dtype, torch.bfloat16)

    # Load base model
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }

    # Try flash attention if specified
    if config.attn_implementation == "flash_attention_2":
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:
            print("Flash Attention 2 not available, using default attention")

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs,
    )

    # Load SFT adapter if it exists
    if config.sft_adapter_path and os.path.exists(config.sft_adapter_path):
        print(f"Loading SFT adapter from {config.sft_adapter_path}")
        base_model = PeftModel.from_pretrained(
            base_model,
            config.sft_adapter_path,
            is_trainable=False,
        )
        base_model = base_model.merge_and_unload()
        print("SFT adapter merged into base model")

    # Apply LoRA for PPO
    if config.use_lora:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        base_model = get_peft_model(base_model, lora_config)
        base_model.print_trainable_parameters()

    # Wrap with value head for PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    return model


def setup_reward_model(config: PPOConfig):
    """Load the trained reward model."""

    print(f"Loading reward model from {config.reward_model_path}")

    # Determine torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(config.torch_dtype, torch.bfloat16)

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
    )
    reward_model.eval()

    reward_tokenizer = AutoTokenizer.from_pretrained(
        config.reward_model_path,
        trust_remote_code=True,
    )

    return reward_model, reward_tokenizer


def prepare_dataset(config: PPOConfig, tokenizer: AutoTokenizer) -> DatasetDict:
    """Load and prepare the prompt dataset for PPO."""

    print("Loading UltraFeedback dataset...")
    raw_dataset = load_ultrafeedback(
        tokenizer=tokenizer,
        max_samples=config.max_samples,
        seed=config.seed,
    )

    print("Converting to PPO format (prompts only)...")
    ppo_dataset = create_ppo_prompts(raw_dataset, tokenizer)

    print(f"Train prompts: {len(ppo_dataset['train'])}")

    return ppo_dataset


def compute_rewards(
    responses: list[str],
    reward_model,
    reward_tokenizer,
    config: PPOConfig,
    device: torch.device,
) -> list[torch.Tensor]:
    """Compute rewards for generated responses."""

    rewards = []
    for response in responses:
        inputs = reward_tokenizer(
            response,
            return_tensors="pt",
            truncation=True,
            max_length=config.max_new_tokens + 512,
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = reward_model(**inputs)
            reward = outputs.logits.squeeze().cpu()

        # Clip reward for stability
        reward = torch.clamp(reward, -config.reward_clip, config.reward_clip)
        rewards.append(reward)

    return rewards


def train_ppo(config: Optional[PPOConfig] = None) -> str:
    """
    Run PPO training.

    Args:
        config: PPOConfig instance. If None, uses defaults.

    Returns:
        Path to the trained model/adapter.
    """
    if config is None:
        config = PPOConfig()

    # Initialize wandb
    if config.report_to == "wandb":
        wandb.init(
            project=config.wandb_project,
            name=config.run_name,
            config=vars(config),
        )

    print("=" * 60)
    print("PPO Training Configuration")
    print("=" * 60)
    print(f"Policy Model: {config.model_name}")
    print(f"SFT Adapter: {config.sft_adapter_path}")
    print(f"Reward Model: {config.reward_model_path}")
    print(f"Output: {config.output_dir}")
    print(f"KL Coefficient: {config.init_kl_coef}")
    print(f"Target KL: {config.target_kl}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    print("=" * 60)

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Setup models
    print("\nLoading policy model...")
    policy_model = setup_policy_model(config, tokenizer)

    print("\nLoading reward model...")
    reward_model, reward_tokenizer = setup_reward_model(config)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare dataset
    dataset = prepare_dataset(config, tokenizer)

    # PPO configuration
    ppo_config = TRLPPOConfig(
        model_name=config.model_name,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        mini_batch_size=config.mini_batch_size,
        ppo_epochs=config.ppo_epochs,
        gamma=config.gamma,
        lam=config.lam,
        cliprange=config.cliprange,
        cliprange_value=config.cliprange_value,
        vf_coef=config.vf_coef,
        init_kl_coef=config.init_kl_coef,
        target_kl=config.target_kl,
        adap_kl_ctrl=config.adap_kl_ctrl,
        horizon=config.horizon,
        use_score_scaling=config.use_score_scaling,
        use_score_norm=config.use_score_norm,
        score_clip=config.score_clip,
        log_with=config.report_to,
        project_kwargs={"project_name": config.wandb_project},
        seed=config.seed,
    )

    # Generation config
    generation_config = GenerationConfig(
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        do_sample=config.do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Initialize PPO trainer
    trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=None,  # Will create copy
        tokenizer=tokenizer,
        dataset=dataset["train"],
    )

    # Training manager for stability monitoring
    manager = PPOTrainingManager(config)

    print("\nStarting PPO training...")
    print(f"Total episodes: {config.total_episodes}")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Training loop
    for episode in range(config.total_episodes):
        manager.step = episode

        try:
            # Sample batch of queries
            batch = trainer.dataloader.__iter__().__next__()
            queries = batch["query"]

            # Generate responses
            query_tensors = [
                tokenizer.encode(q, return_tensors="pt").squeeze().to(device)
                for q in queries
            ]
            response_tensors = trainer.generate(
                query_tensors,
                generation_config=generation_config,
                return_prompt=False,
            )

            # Decode responses
            responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

            # Combine query + response for reward model
            full_responses = [q + r for q, r in zip(queries, responses)]

            # Compute rewards
            rewards = compute_rewards(
                full_responses,
                reward_model,
                reward_tokenizer,
                config,
                device,
            )

            # PPO step
            stats = trainer.step(query_tensors, response_tensors, rewards)

            # Log diagnostics
            diagnostics = manager.log_diagnostics(stats, trainer)

            if episode % config.log_steps == 0:
                print(
                    f"Episode {episode}: "
                    f"reward={diagnostics['reward/mean']:.3f}, "
                    f"kl={diagnostics['kl/mean']:.3f}, "
                    f"entropy={diagnostics['policy/entropy']:.3f}"
                )

                if config.report_to == "wandb":
                    wandb.log(diagnostics, step=episode)

            # Check stability
            is_stable, reason = manager.check_stability(stats)
            if not is_stable:
                print(f"\nWARNING: {reason}")
                print("Consider adjusting KL coefficient or stopping training.")

                # Log warning
                if config.report_to == "wandb":
                    wandb.log({"stability/warning": 1, "stability/reason": reason}, step=episode)

            # Save checkpoint
            if episode > 0 and episode % config.save_steps == 0:
                checkpoint_path = os.path.join(config.output_dir, f"checkpoint-{episode}")
                trainer.save_pretrained(checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

            # Sample outputs for monitoring
            if episode % config.eval_steps == 0 and episode > 0:
                print("\n--- Sample Output ---")
                print(f"Query: {queries[0][:100]}...")
                print(f"Response: {responses[0][:200]}...")
                print(f"Reward: {rewards[0].item():.3f}")
                print("---\n")

        except Exception as e:
            print(f"Error at episode {episode}: {e}")
            if config.report_to == "wandb":
                wandb.log({"error": str(e)}, step=episode)
            continue

    # Save final model
    print("\nSaving final model...")
    trainer.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # Push to hub if configured
    if config.push_to_hub and config.hub_model_id:
        print(f"\nPushing to Hub: {config.hub_model_id}")
        trainer.push_to_hub(config.hub_model_id)

    if config.report_to == "wandb":
        wandb.finish()

    print(f"\nPPO training complete! Model saved to: {config.output_dir}")
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
        reward_model_path: str = typer.Option(
            "./outputs/rm-qwen2.5-1.5b", help="Path to reward model"
        ),
        output_dir: str = typer.Option("./outputs/ppo-qwen2.5-1.5b-lora", help="Output directory"),
        init_kl_coef: float = typer.Option(0.2, help="Initial KL coefficient"),
        target_kl: float = typer.Option(6.0, help="Target KL divergence"),
        learning_rate: float = typer.Option(1e-5, help="Learning rate"),
        batch_size: int = typer.Option(4, help="Batch size"),
        total_episodes: int = typer.Option(10000, help="Total training episodes"),
        lora_r: int = typer.Option(64, help="LoRA rank"),
        lora_alpha: int = typer.Option(128, help="LoRA alpha"),
        max_samples: Optional[int] = typer.Option(None, help="Max samples for debugging"),
        push_to_hub: bool = typer.Option(False, help="Push to HuggingFace Hub"),
        hub_model_id: Optional[str] = typer.Option(None, help="Hub model ID"),
    ):
        """Run PPO training."""
        config = PPOConfig(
            model_name=model_name,
            sft_adapter_path=sft_adapter_path,
            reward_model_path=reward_model_path,
            output_dir=output_dir,
            init_kl_coef=init_kl_coef,
            target_kl=target_kl,
            learning_rate=learning_rate,
            batch_size=batch_size,
            total_episodes=total_episodes,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            max_samples=max_samples,
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
        )
        train_ppo(config)

    app()


if __name__ == "__main__":
    main()
