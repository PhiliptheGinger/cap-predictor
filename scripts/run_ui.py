"""Run the FastAPI UI using Uvicorn."""

from sentimental_cap_predictor.ui.api import app
import uvicorn


def main() -> None:
    """Launch the FastAPI application."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
