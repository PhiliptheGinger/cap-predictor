@echo off
set PYTHONPATH=%~dp0\..\src

python - <<PY %*
from sentimental_cap_predictor.news.fetch_gdelt import search_gdelt
import json, sys
query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "markets"
result = search_gdelt(query)
if result:
    print(result.get("summary", ""))
else:
    print("No article found")
PY
