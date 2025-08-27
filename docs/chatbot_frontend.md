# Chatbot Frontend

This module provides a minimal REPL interface to a Qwen chat model. It
reads configuration from environment variables (loaded with `python-dotenv`) and
communicates with an external runner through a simple `CMD:` protocol.

## Environment variables

The chat frontend looks for the following variables and falls back to
defaults when they are unset:

| Variable | Default | Description |
| --- | --- | --- |
| `QWEN_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | Base URL for the Qwen endpoint |
| `QWEN_API_KEY` | *(empty string)* | API key used to authenticate requests |
| `QWEN_MODEL` | `qwen-max` | Model name passed to the API |
| `LLM_TEMPERATURE` | `0.7` | Sampling temperature |

Variables can be placed in a `.env` file which is loaded automatically.

## Run

```bash
python -m sentimental_cap_predictor.chatbot_frontend
```

## CMD protocol

The assistant may request that the external environment run shell commands by
starting a reply with `CMD:` followed by the command string. The runner detects
these lines, executes the command in a shell, and feeds the captured output back
into the conversation so the model can continue the dialogue.
