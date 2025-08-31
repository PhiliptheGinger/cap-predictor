from __future__ import annotations

from datetime import datetime, timedelta

from sentimental_cap_predictor.config_llm import get_llm_config

SYSTEM_PROMPT = (
    "You are a helpful assistant."
    "\nIf you want me to run a shell command, respond with 'CMD: <command>'."
    "\nFor normal replies, respond without the prefix."
    "\nGDELT articles should be fetched from "
    "https://api.gdeltproject.org/api/v2/doc/doc."
    "\nExample: CMD: curl "
    '"https://api.gdeltproject.org/api/v2/doc/doc?query=ukraine&'
    'mode=ArtList&format=json"'
)


def fetch_gdelt_news(query: str) -> str:
    """Fetch recent news for ``query`` using the GDELT helper.

    A thin wrapper around :func:`dataset.query_gdelt_for_news` that returns the
    raw articles as a JSON string. The helper queries the last 24 hours of
    articles to keep requests lightweight.
    """
    from sentimental_cap_predictor import dataset

    end = datetime.utcnow()
    start = end - timedelta(days=1)
    df = dataset.query_gdelt_for_news(
        query=query,
        start_date=start.strftime("%Y%m%d%H%M%S"),
        end_date=end.strftime("%Y%m%d%H%M%S"),
    )
    return df.to_json(orient="records")


def main() -> None:
    """Run a REPL-style chat session with the local Qwen model."""
    from sentimental_cap_predictor.llm_providers.qwen_local import QwenLocalProvider

    config = get_llm_config()
    provider = QwenLocalProvider(
        model_path=config.model_path, temperature=config.temperature
    )
    history: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    while True:
        try:
            user = input("user> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break

        history.append({"role": "user", "content": user})
        reply = provider.chat(history)
        if reply.startswith("CMD:"):
            output = handle_command(reply[4:].strip())
            print(output)
            history.append({"role": "assistant", "content": output})
        else:
            print(reply)
            history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
