# Qwen Model Weights

The Qwen chatbot requires access to the model weights on the local filesystem.
Provide the location via the `QWEN_MODEL_PATH` environment variable.

## Supplying weights

- **Local path** – Set `QWEN_MODEL_PATH` to a directory containing the
  checkpoint files.
- **Hugging Face repo** – Set `QWEN_MODEL_PATH` to a repository name. The
  weights are downloaded with `huggingface_hub.snapshot_download` before the
  model is loaded.

Optional: set `QWEN_OFFLOAD_FOLDER` to a directory for weights that cannot fit
into GPU/CPU memory.

