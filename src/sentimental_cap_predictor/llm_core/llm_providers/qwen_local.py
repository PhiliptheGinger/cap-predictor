"""Qwen provider using local transformer weights."""

from __future__ import annotations

from typing import Any, List

import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ..utils import clean_generate_kwargs
from .base import ChatMessage, LLMProvider


def _auto_runtime_prefs() -> dict[str, Any]:
    """Return runtime preferences for the local Qwen model.

    The function resolves the model checkpoint location, determines the
    device map, desired dtype and offload directory using environment
    variables.  Any required directories are created automatically.
    """

    import os
    from pathlib import Path

    model_path = os.getenv("QWEN_MODEL_PATH", "Qwen/Qwen2-1.5B-Instruct")
    device_map = os.getenv("QWEN_DEVICE_MAP", "auto")
    dtype_env = os.getenv("QWEN_DTYPE")
    offload_folder = os.getenv("QWEN_OFFLOAD_FOLDER")

    local_model_path = Path(model_path)
    if local_model_path.exists():
        checkpoint_path = model_path
    else:
        checkpoint_path = snapshot_download(
            repo_id=local_model_path.as_posix(),
        )

    offload_dir = (
        Path(offload_folder)
        if offload_folder is not None
        else Path(checkpoint_path) / "offload"
    )
    offload_dir.mkdir(parents=True, exist_ok=True)

    dtype = None
    if dtype_env is not None and dtype_env.lower() != "auto":
        dtype = getattr(torch, dtype_env, None)

    return {
        "model_path": checkpoint_path,
        "device_map": device_map,
        "dtype": dtype,
        "offload_folder": str(offload_dir),
    }


class QwenLocalProvider(LLMProvider):
    """Implementation of :class:`LLMProvider` for a local Qwen model."""

    def __init__(
        self,
        temperature: float,
        max_new_tokens: int = 512,
    ) -> None:
        """Create a provider backed by a local Qwen model.

        Parameters
        ----------
        temperature:
            Sampling temperature for text generation.
        max_new_tokens:
            Default ``max_new_tokens`` value used when chatting with the model.
        """

        prefs = _auto_runtime_prefs()

        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        checkpoint_path = prefs["model_path"]
        self.model_id = checkpoint_path

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        config = AutoConfig.from_pretrained(checkpoint_path)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
        model.tie_weights()

        self.model = load_checkpoint_and_dispatch(
            model,
            checkpoint=checkpoint_path,
            device_map=prefs.get("device_map"),
            offload_folder=prefs.get("offload_folder"),
            dtype=prefs.get("dtype"),
        )
        self.model.eval()

        print(
            f"[runtime] model={self.model_id} "
            f"device_map={prefs.get('device_map')} "
            f"dtype={prefs.get('dtype')} "
            f"offload={prefs.get('offload_folder')}"
        )

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
        generate_kwargs = clean_generate_kwargs(
            temperature=self.temperature,
            **kwargs,
        )
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_kwargs)
        start = inputs["input_ids"].shape[-1]
        generated = outputs[0, start:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)
