import sys
from pathlib import Path
from types import ModuleType

import sentimental_cap_predictor as scp

# Stub out llm_core package to avoid heavy imports
llm_core_path = (
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "sentimental_cap_predictor"
    / "llm_core"
)
llm_core = ModuleType("llm_core")
llm_core.__path__ = [str(llm_core_path)]  # type: ignore[attr-defined]
scp.llm_core = llm_core
sys.modules["sentimental_cap_predictor.llm_core"] = llm_core
