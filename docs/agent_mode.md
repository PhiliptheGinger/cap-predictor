# Agent Mode

Agent mode allows the chat frontend to execute registered tools through a simple
`CMD:` protocol.  It is disabled by default and must be explicitly enabled with
environment variables or command line flags.

## Enabling

1. Enable agent mode and desired tools in your `.env` file:

```env
AGENT_MODE=1
CONFIRM_CMDS=1        # optional safety prompt
ENABLE_WEB_SEARCH=1   # DuckDuckGo web search
ENABLE_PYTHON_RUN=1   # sandboxed python.run tool
ENABLE_FILE_WRITE=1   # file.write tool
ENABLE_READ_URL=1     # read_url tool
ENABLE_MEMORY=1       # memory.upsert and memory.query
```

2. Run the frontend with agent mode:

```bash
python -m sentimental_cap_predictor.llm_core.chatbot_frontend --agent
```

When a tool is invoked the loop logs the command and returns the tool's output
back to the model for further reasoning.  Setting `CONFIRM_CMDS` adds a manual
confirmation step before potentially destructive operations such as writing
files or executing Python code.

## LLM provider

Any supported provider can be used in agent mode.  Select the backend via the
`LLM_PROVIDER` variable or the `--provider` flag.  For example, to use the
DeepSeek API:

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_key_here
```

```bash
python -m sentimental_cap_predictor.llm_core.chatbot_frontend --provider deepseek --agent
```

## Tools

Available built-in tools include:

- `search.web` – DuckDuckGo search
- `python.run` – execute Python code in a sandbox
- `file.write` – write files under `agent_work/`
- `read_url` – fetch and extract text from a URL
- `memory.upsert` / `memory.query` – interact with the vector memory store

Additional tools can register themselves with the agent loop by calling
`register_tool` at import time.

## Demo scripts

The `scripts` directory includes small examples demonstrating the loop:

- `scripts/demo_research_url.py` – fetches and summarizes a web page.
- `scripts/demo_python_function.py` – writes a Python function and runs it.
- `scripts/demo_search_read_memory_code.py` – chains search, reading, memory storage, and code execution.

Run them with `python <script>`.

## Risks and mitigations

Agent mode executes model-specified commands and has inherent risks:

- **Prompt injection**: malicious text may coerce the model into unsafe actions. Keep `CONFIRM_CMDS` enabled and limit available tools.
- **Runaway loops**: poorly behaved prompts could cause endless command cycles. The agent enforces `max_steps` and timeouts to stop runaway behavior.
