# Chatbot Frontend

This module provides a minimal REPL interface to local or remote chat models.
It reads configuration from environment variables (loaded with `python-dotenv`)
and communicates with an external runner through a simple `CMD:` protocol.  An
experimental *agent mode* can register additional tools for the model to call.
By default the frontend runs the legacy Qwen conversation loop; pass `--agent` or set `AGENT_MODE=1` to enable agent mode.

## Environment variables

The chat frontend looks for the following variables and falls back to
defaults when they are unset:

| Variable | Default | Description |
| --- | --- | --- |
| `QWEN_MODEL_PATH` | `Qwen/Qwen2-1.5B-Instruct` | Local path or HF repository for Qwen weights |
| `QWEN_OFFLOAD_FOLDER` | `(unset)` | Directory for weights offloaded from device memory |
| `LLM_TEMPERATURE` | `0.7` | Sampling temperature |
| `LLM_PROVIDER` | `qwen_local` | LLM backend, either `qwen_local` or `deepseek` |
| `DEEPSEEK_MODEL_PATH` | `(unset)` | Local checkpoint for DeepSeek models |
| `DEEPSEEK_API_KEY` | `(unset)` | API key for DeepSeek's hosted service |
| `AGENT_MODE` | `0` | Set to `1` to enable experimental agent loop |
| `CONFIRM_CMDS` | `0` | Require confirmation before executing tool commands |
| `ENABLE_WEB_SEARCH` | `0` | Register DuckDuckGo search tool |
| `ENABLE_PYTHON_RUN` | `0` | Register sandboxed Python execution tool |
| `ENABLE_FILE_WRITE` | `0` | Allow writing files via agent tool |
| `ENABLE_READ_URL` | `0` | Register URL reading tool |
| `ENABLE_MEMORY` | `0` | Register vector memory tools |

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
can be loaded entirely offline. If memory is constrained, use
`QWEN_OFFLOAD_FOLDER` to specify a directory for any weights that need to be
offloaded from the main device.

## Run

```bash
python -m sentimental_cap_predictor.llm_core.chatbot_frontend --provider deepseek --agent --enable-web-search
```

Use `--help` to see all available flags.

## CMD protocol

The assistant may request that the external environment run shell commands by
starting a reply with `CMD:` followed by the command string. The runner detects
these lines, executes the command in a shell, and feeds the captured output back
into the conversation so the model can continue the dialogue.

### Memory search

Articles fetched via `gdelt` commands are automatically embedded and appended to
`data/memory.faiss`, with their titles and URLs recorded in `data/memory.json`.
The `memory search "<query>"` command loads this index of past article content
and performs a similarity search. The handler returns the titles and URLs of the
most relevant matches:

```bash
memory search "climate change policy"
```

If the memory index (`data/memory.faiss`) or accompanying metadata JSON is not
found, a message describing the missing resource is returned instead.

### News commands

Utilities from the news CLI can be accessed through `CMD:` directives:

```bash
CMD: news.fetch_gdelt --query "<terms>" --max 1
CMD: news.read --url "<article_url>" --summarize --analyze --chunks 1000 --json
```

The first command returns matching GDELT articles as JSON. The second retrieves
an article, summarises it, performs a basic analysis and optionally chunks the
text for further processing. When `--summarize` is used on its own, the
response is the summary text; add `--json` to get a JSON object instead.

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
