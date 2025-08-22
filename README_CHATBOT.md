# Chatbot

The project ships with a lightweight command line chatbot for orchestrating
common project actions.

## Launching

Run the bot using the module path:

```bash
python -m sentimental_cap_predictor.chatbot
```

This starts an interactive prompt. Type `help` to see available commands or
`exit` to leave the session.

## Confirmation & Safety

Only handlers defined in the internal registry can be executed. Some actions
(such as shell commands or promoting models) are marked as *dangerous* and
require an explicit confirmation step before dispatch. The parser sets a
`requires_confirmation` flag for these commands and the chat loop prompts with
`Execute?` before calling the handler.

## Example Commands

The natural language parser supports several shortcuts:

- `ingest SPY 1y 1d` – download and prepare price data
- `train model AAPL` – fit baseline models for a ticker
- `compare 1 2` – compare two experiment runs
- `system status` – report Python and platform versions
- `shell ls` – run a shell command (requires confirmation)

## Extending With New Commands

1. Implement a handler function that performs the desired action and returns a
   message or mapping.
2. Register it in `agent/command_registry.get_registry` by adding a
   `Command` entry with a unique name, summary, parameter schema and optional
   `dangerous=True` flag.
3. Teach the parser how to invoke the command by adding a pattern to
   `agent/nl_parser.parse`. If the action is dangerous set
   `requires_confirmation=True` in the returned `Intent`.
4. Launch the chatbot and invoke the new command from the prompt.

The dispatcher automatically validates parameters against the provided schema
using Pydantic before running the handler.
