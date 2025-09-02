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

Install the required libraries.  The local provider relies on
`transformers`, `accelerate`, and `safetensors` in addition to `torch`:

```bash
pip install transformers accelerate safetensors torch
```

If you need an offline setup, download the wheels ahead of time and
install from the local directory:

```bash
pip download transformers accelerate safetensors torch -d ./wheels
pip install --no-index --find-links=./wheels transformers accelerate safetensors torch
```

Download the Qwen weights (for example using the Hugging Face CLI):

```bash
huggingface-cli download Qwen/Qwen2-1.5B-Instruct --local-dir /path/to/qwen
```

Set `QWEN_MODEL_PATH` to the directory containing the weights so the model
can be loaded entirely offline.

## Run

```bash
python -m sentimental_cap_predictor.chatbot_frontend
```

## CMD protocol

The assistant may request that the external environment run shell commands by
starting a reply with `CMD:` followed by the command string. The runner detects
these lines, executes the command in a shell, and feeds the captured output back
into the conversation so the model can continue the dialogue.

### Memory search

The `memory search "<query>"` command loads a local FAISS index of past article
content and performs a similarity search. The handler returns the titles and
URLs of the most relevant matches:

```bash
memory search "climate change policy"
```

If the memory index (`data/memory.faiss`) or accompanying metadata JSON is not
found, a message describing the missing resource is returned instead.

## Connector intents

The chatbot understands simple phrases to pull data from a few external
sources:

- `arxiv machine learning`
- `pubmed cancer research`
- `openalex reinforcement learning`
- `fred GDP`
- `github repo openai/gpt-4`

Each phrase maps to an intent (e.g. `info.arxiv`, `info.fred`) which invokes the
corresponding connector helper.
