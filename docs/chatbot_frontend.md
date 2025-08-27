# Chatbot Frontend

This module provides a minimal REPL interface to a **local** Qwen chat model.
It reads configuration from environment variables (loaded with `python-dotenv`)
and communicates with an external runner through a simple `CMD:` protocol.

## Environment variables

The chat frontend looks for the following variables and falls back to
defaults when they are unset:

| Variable | Default | Description |
| --- | --- | --- |
| `QWEN_MODEL_PATH` | `Qwen/Qwen2-1.5B-Instruct` | Local path or HF repository for the weights |
| `LLM_TEMPERATURE` | `0.7` | Sampling temperature |

Variables can be placed in a `.env` file which is loaded automatically.

## Dependencies and weights

Install the required libraries:

```bash
pip install transformers torch
```

Download the Qwen weights (for example using the Hugging Face CLI):

```bash
huggingface-cli download Qwen/Qwen2-1.5B-Instruct --local-dir /path/to/qwen
```

Set `QWEN_MODEL_PATH` to the directory containing the weights.

## Run

```bash
python -m sentimental_cap_predictor.chatbot_frontend
```

## CMD protocol

The assistant may request that the external environment run shell commands by
starting a reply with `CMD:` followed by the command string. The runner detects
these lines, executes the command in a shell, and feeds the captured output back
into the conversation so the model can continue the dialogue.
