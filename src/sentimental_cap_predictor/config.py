import os
import sys
from pathlib import Path
import logging

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    def load_dotenv(*_args, **_kwargs):  # type: ignore[override]
        """Fallback no-op if python-dotenv is not installed."""

try:  # pragma: no cover - optional dependency
    from loguru import logger  # type: ignore
    _HAS_LOGURU = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _HAS_LOGURU = False
    logger = logging.getLogger("sentimental_cap_predictor")

# Load environment variables from .env file
load_dotenv()

# Paths
PACKAGE_DIR = Path(__file__).resolve().parent
PROJ_ROOT = PACKAGE_DIR.parents[1]


def log_project_root() -> None:
    """Log the resolved project root path."""
    logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")


DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", DATA_DIR / "raw"))
INTERIM_DATA_DIR = Path(
    os.getenv("INTERIM_DATA_DIR", DATA_DIR / "interim"),
)
PROCESSED_DATA_DIR = Path(
    os.getenv("PROCESSED_DATA_DIR", DATA_DIR / "processed"),
)
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
MODELING_DIR = Path(os.getenv("MODELING_DIR", PACKAGE_DIR / "modeling"))
TRADER_DIR = PACKAGE_DIR / "trading" / "trader_utils"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# API endpoints
GDELT_API_URL = os.getenv(
    "GDELT_API_URL", "https://api.gdeltproject.org/api/v2/doc/doc"
)

# Model hyperparameters
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.001))
LNN_UNITS = int(os.getenv("LNN_UNITS", 64))
DROPOUT_RATE = float(os.getenv("DROPOUT_RATE", 0.2))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", 10))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
EPOCHS = int(os.getenv("EPOCHS", 50))
TRAIN_RATIO = float(os.getenv("TRAIN_RATIO", 0.8))

# SARIMA model parameters
SARIMA_ORDER = tuple(map(int, os.getenv("SARIMA_ORDER", "1,1,1").split(",")))
SARIMA_SEASONAL_ORDER = tuple(
    map(int, os.getenv("SARIMA_SEASONAL_ORDER", "1,1,1,7").split(","))
)
SEASONAL_ORDER = tuple(
    map(int, os.getenv("SEASONAL_ORDER", "1,1,1,12").split(",")),
)

# FFT Analysis
PREDICTION_DAYS = int(os.getenv("PREDICTION_DAYS", 14))
TRAIN_SIZE_RATIO = float(os.getenv("TRAIN_SIZE_RATIO", 0.8))

# Data path
DATA_PATH = os.getenv("DATA_PATH", "./data/your_data.csv")

# Logging level
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Toggle per‑ticker logging of pipeline runs
ENABLE_TICKER_LOGS = os.getenv("ENABLE_TICKER_LOGS", "0") == "1"

# Configure logging.  When loguru is available we keep the previous behaviour,
# otherwise fall back to the standard ``logging`` module.
if _HAS_LOGURU:
    try:
        logger.remove()
    except ValueError:
        pass
    stdout_handler_id = logger.add(sys.stdout, level=LOG_LEVEL)
    try:
        from tqdm import tqdm  # type: ignore

        logger.remove(stdout_handler_id)
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        pass
else:  # pragma: no cover - simple logging configuration
    logging.basicConfig(level=LOG_LEVEL)

# Ticker List from .env file
TICKER_LIST_TECH = os.getenv("TICKER_LIST_TECH", "").split(",")
TICKER_LIST_ENERGY = os.getenv("TICKER_LIST_ENERGY", "").split(",")
TICKER_LIST_HEALTH = os.getenv("TICKER_LIST_HEALTH", "").split(",")
TICKER_LIST_AUTO = os.getenv("TICKER_LIST_AUTO", "").split(",")
TICKER_LIST_FINANCIAL = os.getenv("TICKER_LIST_FINANCIAL", "").split(",")
TICKER_LIST_MISC = os.getenv("TICKER_LIST_MISC", "").split(",")

# Combine all ticker lists
TICKER_LIST = (
    TICKER_LIST_TECH
    + TICKER_LIST_ENERGY
    + TICKER_LIST_HEALTH
    + TICKER_LIST_AUTO
    + TICKER_LIST_FINANCIAL
    + TICKER_LIST_MISC
)

# Clean up any extra spaces or empty strings
TICKER_LIST = [ticker.strip() for ticker in TICKER_LIST if ticker.strip()]

if ENABLE_TICKER_LOGS:
    logger.info(f"Final ticker list: {TICKER_LIST}")


if __name__ == "__main__":
    log_project_root()
