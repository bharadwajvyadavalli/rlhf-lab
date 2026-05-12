"""
Gradio Demo: Side-by-Side Model Comparison

Compare outputs from SFT, DPO, and PPO models on the same prompt.
Optionally shows reward model scores for each response.
"""

import os
from typing import Optional

import gradio as gr
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
)


class ModelManager:
    """Manages loading and inference for multiple models."""

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-1.5B",
        sft_adapter_path: Optional[str] = "./outputs/sft-qwen2.5-1.5b-lora",
        dpo_adapter_path: Optional[str] = "./outputs/dpo-qwen2.5-1.5b-lora",
        ppo_adapter_path: Optional[str] = "./outputs/ppo-qwen2.5-1.5b-lora",
        reward_model_path: Optional[str] = "./outputs/rm-qwen2.5-1.5b",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
    ):
        self.base_model_name = base_model_name
        self.sft_adapter_path = sft_adapter_path
        self.dpo_adapter_path = dpo_adapter_path
        self.ppo_adapter_path = ppo_adapter_path
        self.reward_model_path = reward_model_path
        self.device = device

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.torch_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        self.models = {}
        self.tokenizer = None
        self.reward_model = None
        self.reward_tokenizer = None

    def load_tokenizer(self):
        """Load the tokenizer."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, name: str, adapter_path: Optional[str] = None):
        """Load a model with optional adapter."""
        if name in self.models:
            return self.models[name]

        print(f"Loading {name} model...")

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        if adapter_path and os.path.exists(adapter_path):
            print(f"  Loading adapter from {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()

        model.eval()
        self.models[name] = model
        print(f"  {name} loaded successfully")

        return model

    def load_all_models(self):
        """Load all models for comparison."""
        self.load_tokenizer()

        # Load each model variant
        self.load_model("sft", self.sft_adapter_path)
        self.load_model("dpo", self.dpo_adapter_path)
        self.load_model("ppo", self.ppo_adapter_path)

        # Load reward model if available
        if self.reward_model_path and os.path.exists(self.reward_model_path):
            print("Loading reward model...")
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                self.reward_model_path,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            self.reward_model.eval()
            self.reward_tokenizer = AutoTokenizer.from_pretrained(
                self.reward_model_path,
                trust_remote_code=True,
            )
            if self.reward_tokenizer.pad_token is None:
                self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token
            print("  Reward model loaded")

    def generate(
        self,
        model_name: str,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate response from a specific model."""
        model = self.models.get(model_name)
        if model is None:
            return f"Model {model_name} not loaded"

        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(formatted, return_tensors="pt").to(model.device)

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=generation_config)

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return response.strip()

    def get_reward(self, prompt: str, response: str) -> Optional[float]:
        """Get reward score for a response."""
        if self.reward_model is None:
            return None

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        text = self.reward_tokenizer.apply_chat_template(messages, tokenize=False)

        inputs = self.reward_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.reward_model.device)

        with torch.no_grad():
            outputs = self.reward_model(**inputs)
            reward = outputs.logits.squeeze().item()

        return reward


# Global model manager
manager = None


def initialize_models():
    """Initialize the model manager."""
    global manager
    if manager is None:
        manager = ModelManager()
        manager.load_all_models()
    return manager


def compare_models(
    prompt: str,
    max_tokens: int,
    temperature: float,
    show_rewards: bool,
):
    """Generate and compare responses from all models."""
    global manager

    if manager is None:
        initialize_models()

    # Generate responses
    sft_response = manager.generate("sft", prompt, max_tokens, temperature)
    dpo_response = manager.generate("dpo", prompt, max_tokens, temperature)
    ppo_response = manager.generate("ppo", prompt, max_tokens, temperature)

    # Get rewards if requested
    if show_rewards and manager.reward_model is not None:
        sft_reward = manager.get_reward(prompt, sft_response)
        dpo_reward = manager.get_reward(prompt, dpo_response)
        ppo_reward = manager.get_reward(prompt, ppo_response)

        sft_response = f"{sft_response}\n\n---\nReward Score: {sft_reward:.3f}"
        dpo_response = f"{dpo_response}\n\n---\nReward Score: {dpo_reward:.3f}"
        ppo_response = f"{ppo_response}\n\n---\nReward Score: {ppo_reward:.3f}"

    return sft_response, dpo_response, ppo_response


def create_demo():
    """Create the Gradio demo interface."""

    with gr.Blocks(title="RLHF/DPO Model Comparison", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # RLHF/DPO Model Comparison

            Compare responses from models trained with different methods:
            - **SFT**: Supervised Fine-Tuning baseline
            - **DPO**: Direct Preference Optimization
            - **PPO**: Proximal Policy Optimization with Reward Model

            Enter a prompt below to see side-by-side comparisons.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=3,
                )

            with gr.Column(scale=1):
                max_tokens = gr.Slider(
                    minimum=64,
                    maximum=1024,
                    value=512,
                    step=64,
                    label="Max Tokens",
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.5,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                )
                show_rewards = gr.Checkbox(
                    label="Show Reward Scores",
                    value=True,
                )

        generate_btn = gr.Button("Compare Models", variant="primary")

        with gr.Row():
            sft_output = gr.Textbox(
                label="SFT (Baseline)",
                lines=15,
                show_copy_button=True,
            )
            dpo_output = gr.Textbox(
                label="DPO",
                lines=15,
                show_copy_button=True,
            )
            ppo_output = gr.Textbox(
                label="PPO",
                lines=15,
                show_copy_button=True,
            )

        # Example prompts
        gr.Examples(
            examples=[
                ["Explain the concept of machine learning to a 10-year-old."],
                ["Write a short poem about the beauty of mathematics."],
                ["What are the key differences between supervised and unsupervised learning?"],
                ["Help me debug this Python code: def fibonacci(n): return fibonacci(n-1) + fibonacci(n-2)"],
                ["Summarize the main causes of climate change in 3 bullet points."],
            ],
            inputs=prompt_input,
        )

        generate_btn.click(
            fn=compare_models,
            inputs=[prompt_input, max_tokens, temperature, show_rewards],
            outputs=[sft_output, dpo_output, ppo_output],
        )

        gr.Markdown(
            """
            ---
            **Note**: Models are loaded on first use. Initial response may take longer.

            Built as part of the RLHF/DPO Bridging Project.
            """
        )

    return demo


def main():
    """Launch the Gradio demo."""
    import argparse

    parser = argparse.ArgumentParser(description="RLHF/DPO Model Comparison Demo")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-1.5B",
        help="Base model name",
    )
    parser.add_argument(
        "--sft-adapter",
        default="./outputs/sft-qwen2.5-1.5b-lora",
        help="SFT adapter path",
    )
    parser.add_argument(
        "--dpo-adapter",
        default="./outputs/dpo-qwen2.5-1.5b-lora",
        help="DPO adapter path",
    )
    parser.add_argument(
        "--ppo-adapter",
        default="./outputs/ppo-qwen2.5-1.5b-lora",
        help="PPO adapter path",
    )
    parser.add_argument(
        "--reward-model",
        default="./outputs/rm-qwen2.5-1.5b",
        help="Reward model path",
    )

    args = parser.parse_args()

    # Initialize with custom paths
    global manager
    manager = ModelManager(
        base_model_name=args.base_model,
        sft_adapter_path=args.sft_adapter,
        dpo_adapter_path=args.dpo_adapter,
        ppo_adapter_path=args.ppo_adapter,
        reward_model_path=args.reward_model,
    )

    print("Starting Gradio demo...")
    print("Models will be loaded on first request.")

    demo = create_demo()
    demo.launch(
        share=args.share,
        server_port=args.port,
    )


if __name__ == "__main__":
    main()
