import sys
from pathlib import Path

# Ensure the src directory is importable for tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
