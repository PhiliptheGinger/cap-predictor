"""Qwen provider using local transformer weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .base import ChatMessage, LLMProvider


class QwenLocalProvider(LLMProvider):
    """Implementation of :class:`LLMProvider` for a local Qwen model."""

    def __init__(
        self,
        model_path: str,
        temperature: float,
        max_new_tokens: int = 512,
        offload_folder: str | None = None,
    ) -> None:
        """Create a provider backed by a local Qwen model.

        Parameters
        ----------
        model_path:
            Path to the model weights on disk or a Hugging Face repository
            name.
        temperature:
            Sampling temperature for text generation.
        max_new_tokens:
            Default ``max_new_tokens`` value used when chatting with the model.
        offload_folder:
            Directory used by ``accelerate`` to offload model weights when the
            full model cannot fit in device memory. If ``None`` the model is
            loaded entirely on the GPU/CPU indicated by ``device_map``.
        """

        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        local_model_path = Path(model_path)
        if local_model_path.exists():
            checkpoint_path = model_path
        else:
            checkpoint_path = snapshot_download(
                repo_id=local_model_path.as_posix(),
            )

        # Debug print to show the resolved checkpoint path. The previous
        # implementation attempted to reference ``model_id`` which does not
        # exist in this context. Use ``checkpoint_path`` instead so the
        # message reflects the actual path used to load the model weights.
        print(f"Using checkpoint at: {checkpoint_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        config = AutoConfig.from_pretrained(checkpoint_path)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
        model.tie_weights()
        self.model = load_checkpoint_and_dispatch(
            model,
            checkpoint=checkpoint_path,
            device_map="auto",
            offload_folder=offload_folder,
        )
        self.model.eval()

    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> str:
        """Return the model's response to a list of chat messages."""

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokenized = self.tokenizer(prompt, return_tensors="pt")
        inputs = tokenized.to(self.model.device)
        kwargs.setdefault("max_new_tokens", self.max_new_tokens)
        kwargs.setdefault(
            "max_length",
            inputs["input_ids"].shape[-1] + kwargs["max_new_tokens"],
        )
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, temperature=self.temperature, **kwargs
            )
        start = inputs["input_ids"].shape[-1]
        generated = outputs[0, start:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)
