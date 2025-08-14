import sys
from pathlib import Path

# Ensure the src directory is on the Python path so tests can import the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
